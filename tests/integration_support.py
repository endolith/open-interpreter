"""Hooks for integration tests that call an LLM and execute generated code."""

import hashlib
import os
import sys

import pytest

_approve_all = False
_approved_hashes: set[str] = set()


def integration_tests_allowed() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    # CI runners are isolated; local runs need an explicit opt-in.
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return True
    return os.environ.get("OI_RUN_INTEGRATION") == "1"


def auto_approve_integration_code() -> bool:
    return os.environ.get("OI_AUTO_APPROVE_INTEGRATION") == "1"


def _code_hash(language: str, code: str) -> str:
    return hashlib.sha256(f"{language}\0{code}".encode()).hexdigest()


def _tty_print(msg: str) -> None:
    # pytest replaces sys.stderr/stdout; sys.__stderr__ is the original fd and
    # goes directly to the terminal regardless of capture settings.
    sys.__stderr__.write(msg + "\n")
    sys.__stderr__.flush()


def _tty_input(prompt: str) -> str:
    # sys.__stdin__ is the original stdin fd, before pytest's capture wrapper.
    sys.__stderr__.write(prompt)
    sys.__stderr__.flush()
    line = sys.__stdin__.readline()
    return line.rstrip("\n")


def prompt_for_code_execution(*, test_name: str, language: str, code: str) -> bool:
    global _approve_all

    if auto_approve_integration_code() or _approve_all:
        return True

    digest = _code_hash(language, code)
    if digest in _approved_hashes:
        return True

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

    # sys.__stdin__ is the real terminal even under pytest; isatty() on
    # sys.stdin would return False because pytest wraps it for capture.
    if not sys.__stdin__.isatty():
        pytest.fail(
            "Integration test wants to execute LLM-generated code in a non-interactive "
            "session. Re-run in a terminal to approve each block, or set "
            "OI_AUTO_APPROVE_INTEGRATION=1 if you accept the risk."
        )

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
    than patching Terminal.run, so display=True / terminal_interface routing is
    completely unaffected.
    """
    from interpreter.core import core as core_mod

    original_chat = core_mod.OpenInterpreter.chat

    def approving_chat(self, message=None, display=False, stream=False, blocking=True):
        raw = original_chat(self, message=message, display=False, stream=True, blocking=blocking)

        def guarded():
            for chunk in raw:
                if chunk.get("type") == "confirmation":
                    language = chunk.get("content", {}).get("format", "?")
                    code = chunk.get("content", {}).get("content", "")
                    if not prompt_for_code_execution(
                        test_name=test_name, language=language, code=code
                    ):
                        pytest.fail(
                            f"LLM-generated {language} code was not approved for {test_name!r}"
                        )
                yield chunk

        if stream:
            return guarded()

        # Non-streaming: collect into a list, just like the real chat(stream=False)
        output_messages = []
        for chunk in guarded():
            if chunk.get("format") != "active_line":
                if (
                    output_messages
                    and output_messages[-1].get("type") == chunk.get("type")
                    and output_messages[-1].get("format") == chunk.get("format")
                    and "start" not in chunk
                    and "end" not in chunk
                ):
                    output_messages[-1]["content"] += chunk.get("content", "")
                else:
                    output_messages.append(chunk)
        return output_messages

    monkeypatch.setattr(core_mod.OpenInterpreter, "chat", approving_chat)
