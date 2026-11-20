"""Hooks for integration tests that call an LLM and execute generated code."""

import hashlib
import os
import sys
import threading

import pytest

_approve_all = False
_approved_hashes: set[str] = set()
_TERMINAL_PATH = "CON" if os.name == "nt" else "/dev/tty"
_terminal = None  # cached (reader, writer) pair, or False when unavailable

_NO_TERMINAL = (
    "Integration tests execute LLM-generated code, so they ask for approval first, "
    "but this session has no interactive terminal. Run pytest from a terminal window, "
    "or set OI_AUTO_APPROVE_INTEGRATION=1 if you accept the risk."
)


def _open_terminal(mode: str):
    """Open the controlling terminal in the given text mode."""
    return open(_TERMINAL_PATH, mode)


def _terminal_handles():
    """Return a (reader, writer) pair on the controlling terminal, or None.

    pytest's default `--capture=fd` dup2()s fds 0-2 onto temp files, so
    `sys.__stdin__` and `sys.__stderr__` are captured too; a prompt has to open
    the terminal itself to reach the user. The two directions get separate
    handles because a tty is not seekable: the text-mode update handle
    `open(path, "r+")` raises `io.UnsupportedOperation` on one.
    """
    global _terminal
    if _terminal is None:
        reader = writer = None
        try:
            reader = _open_terminal("r")
            writer = _open_terminal("w")
        except OSError:
            if reader is not None:
                reader.close()
            _terminal = False
        else:
            _terminal = (reader, writer)
    return _terminal or None


def terminal_available() -> bool:
    """Return True when a controlling terminal can be opened for prompts."""
    return _terminal_handles() is not None


def auto_approve_integration_code() -> bool:
    """Return True when the user asked to skip the per-block approval prompt."""
    return os.environ.get("OI_AUTO_APPROVE_INTEGRATION") == "1"


def _code_hash(language: str, code: str) -> str:
    """Return a stable key for a block so repeated approvals are remembered."""
    return hashlib.sha256(f"{language}\0{code}".encode()).hexdigest()


_ESCAPED_CONTROL_CHARS = {
    code: f"\\x{code:02x}"
    for code in (*range(0x20), *range(0x7F, 0xA0))
    if code not in (0x09, 0x0A)
}
"""Control bytes rendered inert for display; tabs and newlines stay as layout."""


def _escape_terminal_control_characters(text: str) -> str:
    """Return `text` with control bytes escaped so a preview can only render text.

    The block shown at the prompt comes from the model, so an escape sequence
    such as ``\\x1b[2J`` would otherwise clear the screen or move the cursor and
    let generated code rewrite the question the user is answering. Newlines and
    tabs are kept because the prompt is laid out with them; hashing and
    execution use the original, unescaped code.
    """
    return text.translate(_ESCAPED_CONTROL_CHARS)


def _tty_print(msg: str) -> None:
    msg = _escape_terminal_control_characters(msg)
    handles = _terminal_handles()
    if handles is None:
        # No terminal to write to: fall back to stderr so the text still lands
        # in the captured output for diagnosis.
        sys.__stderr__.write(msg + "\n")
        sys.__stderr__.flush()
        return
    writer = handles[1]
    writer.write(msg + "\n")
    writer.flush()


def _tty_input(prompt: str) -> str:
    handles = _terminal_handles()
    if handles is None:
        pytest.fail(_NO_TERMINAL)
    reader, writer = handles
    writer.write(prompt)
    writer.flush()
    line = reader.readline()
    if not line:
        pytest.fail(_NO_TERMINAL)
    return line.rstrip("\n")


def prompt_for_code_execution(*, test_name: str, language: str, code: str) -> bool:
    """Ask before running an LLM-generated block; True means run it.

    Answers are normalized to y/n/a, "a" approves the rest of the session, and a
    block that was approved once is not asked about again.
    """
    global _approve_all

    if auto_approve_integration_code() or _approve_all:
        return True

    digest = _code_hash(language, code)
    if digest in _approved_hashes:
        return True

    if not terminal_available():
        pytest.fail(_NO_TERMINAL)

    lines = [
        "",
        "=" * 72,
        f"  [INTEGRATION] {test_name}  —  language: {language}",
        "-" * 72,
        code.rstrip(),
        "=" * 72,
        "  Run this code? [y]es / [n]o / [a]ll",
        "",
    ]
    _tty_print("\n".join(lines))

    while True:
        answer = _tty_input("> ").strip().lower()
        if answer in {"y", "yes"}:
            _approved_hashes.add(digest)
            return True
        if answer in {"n", "no", ""}:
            return False
        if answer in {"a", "all"}:
            _approve_all = True
            return True
        _tty_print("  Please enter y, n, or a.")


def install_chat_approval(monkeypatch, test_name: str) -> None:
    """Gate integration runs so no LLM-generated block executes unapproved.

    The prompt sits at ``Computer.run``, the seam where ``respond.py`` hands the
    model's code to the terminal, rather than at the ``confirmation`` chunk.
    ``core.py`` drops that chunk whenever ``auto_run`` is True - which
    ``tests/test_interpreter.py`` sets - so intercepting it let code run with no
    prompt at all.

    ``chat`` is pinned to ``auto_run=True`` while it runs so ``terminal_interface``
    does not ask a second time for the same block: with ``auto_run`` False it
    receives the confirmation chunk and prompts, and the execution guard would
    then prompt again.
    """
    _install_chat_approval(monkeypatch.setattr, test_name)


def install_process_approval(test_name: str) -> None:
    """Install the guard for the rest of this process, for a spawned server.

    ``test_server`` runs the interpreter in a spawned child, and a child does not
    inherit the pytest process's monkeypatches: without this the server would
    execute the model's code with no prompt while the parent test believes it is
    guarded. There is nothing to undo - the child exits with its server.
    """
    _install_chat_approval(setattr, test_name)


def _install_chat_approval(set_attribute, test_name: str) -> None:
    """Patch the execution seam and ``chat`` through ``set_attribute``."""
    from interpreter.core import core as core_mod
    from interpreter.core.computer.computer import Computer

    original_chat = core_mod.OpenInterpreter.chat
    original_wait = core_mod.OpenInterpreter.wait
    original_run = Computer.run

    def guarded_run(self, *args, **kwargs):
        """Prompt before running the block `respond.py` took from a reply.

        Only the model's block is passed ``stream=True`` by ``respond.py``
        (``interpreter.computer.run(language, code, stream=True)``). Internal
        calls - the computer-API import and ``sync_computer`` - omit it and run
        trusted scaffolding, so they must not prompt.
        """
        language = args[0] if args else kwargs.get("language", "?")
        code = args[1] if len(args) > 1 else kwargs.get("code", "")
        stream = args[2] if len(args) > 2 else kwargs.get("stream", False)
        if stream and not prompt_for_code_execution(
            test_name=test_name, language=language, code=code
        ):
            pytest.fail(
                f"LLM-generated {language} code was not approved for {test_name!r}"
            )
        return original_run(self, *args, **kwargs)

    set_attribute(Computer, "run", guarded_run)

    def run_worker(self, message, display, stream):
        """Run a non-blocking turn in a thread whose failure ``wait`` can raise.

        Delegating ``blocking=False`` to the original starts the thread inside
        ``core.py``, out of reach: a refusal raises there, ``chat`` clears
        ``responding`` while unwinding, and ``wait`` then returns normally, so
        the denial reaches pytest only as an unhandled-thread warning instead of
        a failure. Owning the thread here lets its exception be stored on the
        instance for ``checked_wait`` to re-raise.
        """
        threads = self.__dict__.setdefault("_integration_worker_threads", [])
        failures = self.__dict__.setdefault("_integration_worker_failures", [])

        def worker():
            """Run the turn in this thread, keeping any failure for ``wait``."""
            try:
                self.chat(
                    message=message, display=display, stream=stream, blocking=True
                )
            except BaseException as exc:
                # ``wait`` cannot catch an exception raised in ``core.py``'s
                # own thread, so record it for ``checked_wait`` to re-raise.
                failures.append(exc)

        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()

    def checked_wait(self):
        """Join the harness's worker turns, then re-raise a stored failure.

        Joining first also closes the race where ``chat`` clears ``responding``
        before the worker records the exception: ``wait`` would otherwise return
        in between and never see the denial.
        """
        threads = self.__dict__.get("_integration_worker_threads", [])
        while threads:
            threads.pop().join()
        failures = self.__dict__.get("_integration_worker_failures", [])
        if failures:
            raise failures.pop(0)
        return original_wait(self)

    set_attribute(core_mod.OpenInterpreter, "wait", checked_wait)

    def approving_chat(self, message=None, display=True, stream=False, blocking=True):
        """Run the real chat with ``auto_run`` pinned, restoring it afterwards.

        The execution guard is the only approval prompt. Under ``display=True``
        the real chat routes through ``terminal_interface``, which prompts on the
        confirmation chunk when ``auto_run`` is False; pinning it True keeps that
        second prompt from firing. A ``blocking=False`` call starts a
        harness-owned worker thread that re-enters this wrapper with
        ``blocking=True``, which does the pinning for the call that actually
        streams; owning the thread is what lets ``wait`` re-raise a refusal it
        hit.
        """
        if not blocking:
            run_worker(self, message, display, stream)
            return

        previous = self.auto_run

        def restore():
            """Put the interpreter's original ``auto_run`` value back."""
            self.auto_run = previous

        if stream:

            def pinned(generator):
                """Pin ``auto_run`` while the stream runs, then restore it.

                ``chat(stream=True)`` only builds the generator, so pinning at
                call time left ``auto_run`` True for a stream the caller never
                iterated - its ``finally`` never ran. Pinning on first iteration
                keeps ``terminal_interface`` from asking a second time for the
                same block while the stream is live, and an unused stream that
                is closed again leaves the previous value alone.
                """
                self.auto_run = True
                try:
                    yield from generator
                finally:
                    restore()

            return pinned(
                original_chat(
                    self, message=message, display=display, stream=True, blocking=True
                )
            )

        self.auto_run = True
        try:
            return original_chat(
                self, message=message, display=display, stream=False, blocking=True
            )
        finally:
            restore()

    set_attribute(core_mod.OpenInterpreter, "chat", approving_chat)
