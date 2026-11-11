"""Hooks for integration tests that call an LLM and execute generated code."""

import hashlib
import os
import sys

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


def _tty_print(msg: str) -> None:
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


def install_chat_approval(monkeypatch, test_name: str):
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
    from interpreter.core import core as core_mod
    from interpreter.core.computer.computer import Computer

    original_chat = core_mod.OpenInterpreter.chat
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

    monkeypatch.setattr(Computer, "run", guarded_run)

    def approving_chat(self, message=None, display=True, stream=False, blocking=True):
        """Run the real chat with ``auto_run`` pinned, restoring it afterwards.

        The execution guard is the only approval prompt. Under ``display=True``
        the real chat routes through ``terminal_interface``, which prompts on the
        confirmation chunk when ``auto_run`` is False; pinning it True keeps that
        second prompt from firing. A ``blocking=False`` call returns immediately
        and its worker thread re-enters this wrapper with ``blocking=True``, which
        does the pinning for the call that actually streams.
        """
        if not blocking:
            return original_chat(
                self, message=message, display=display, stream=stream, blocking=False
            )

        previous = self.auto_run
        self.auto_run = True

        def restore():
            """Put the interpreter's original ``auto_run`` value back."""
            self.auto_run = previous

        try:
            result = original_chat(
                self, message=message, display=display, stream=stream, blocking=True
            )
        except BaseException:
            restore()
            raise

        if not stream:
            restore()
            return result

        def pinned(generator):
            """Forward the chunk stream, restoring ``auto_run`` when it closes."""
            try:
                yield from generator
            finally:
                restore()

        return pinned(result)

    monkeypatch.setattr(core_mod.OpenInterpreter, "chat", approving_chat)
