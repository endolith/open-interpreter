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


def integration_tests_allowed() -> bool:
    """Return True when the integration suite has consent to run and can run."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    # CI runners are isolated; local runs need an explicit opt-in.
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return True
    return os.environ.get("OI_RUN_INTEGRATION") == "1"


def auto_approve_integration_code() -> bool:
    """Return True when the user asked to skip the per-block approval prompt."""
    return os.environ.get("OI_AUTO_APPROVE_INTEGRATION") == "1"


def _code_hash(language: str, code: str) -> str:
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
    """Wrap interpreter.chat to intercept confirmation chunks before code runs.

    respond.py yields a {"type": "confirmation"} chunk exactly when LLM-generated
    code is about to execute. We intercept that chunk from the chat stream rather
    than patching Terminal.run, so the display= routing stays untouched.
    """
    from interpreter.core import core as core_mod

    original_chat = core_mod.OpenInterpreter.chat

    def approving_chat(self, message=None, display=True, stream=False, blocking=True):
        if display or not blocking:
            # Keep the real routing: display=True goes through terminal_interface,
            # and blocking=False returns after starting a worker thread that
            # re-enters chat(display=False, stream=True, blocking=True) below.
            return original_chat(
                self, message=message, display=display, stream=stream, blocking=blocking
            )

        raw = original_chat(self, message=message, display=False, stream=True)

        def guarded():
            for chunk in raw:
                if chunk.get("type") == "confirmation":
                    language = chunk.get("content", {}).get("format", "?")
                    code = chunk.get("content", {}).get("content", "")
                    if not prompt_for_code_execution(test_name=test_name, language=language, code=code):
                        pytest.fail(f"LLM-generated {language} code was not approved for {test_name!r}")
                yield chunk

        if stream:
            return guarded()

        # Non-streaming: drain the guarded stream, then mirror chat(stream=False):
        # it resets responding and returns the messages stored by _streaming_chat.
        for _ in guarded():
            pass
        self.responding = False
        return self.messages[self.last_messages_count :]

    monkeypatch.setattr(core_mod.OpenInterpreter, "chat", approving_chat)
