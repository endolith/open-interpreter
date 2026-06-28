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


def prompt_for_code_execution(*, test_name: str, language: str, code: str) -> bool:
    global _approve_all

    if auto_approve_integration_code() or _approve_all:
        return True

    digest = _code_hash(language, code)
    if digest in _approved_hashes:
        return True

    banner = (
        f"\n{'=' * 72}\n"
        f"Integration test: {test_name}\n"
        f"Language: {language}\n"
        f"The LLM produced code that Open Interpreter is about to run on your machine.\n"
        f"{'-' * 72}\n"
        f"{code.rstrip()}\n"
        f"{'=' * 72}"
    )
    print(banner, flush=True)

    if not sys.stdin.isatty():
        pytest.fail(
            "Integration test wants to execute LLM-generated code in a non-interactive "
            "session. Re-run in a terminal to approve each block, or set "
            "OI_AUTO_APPROVE_INTEGRATION=1 if you accept the risk."
        )

    while True:
        answer = input("Execute this code? [y]es / [n]o / [a]pprove all remaining: ").strip().lower()
        if answer in {"y", "yes"}:
            _approved_hashes.add(digest)
            return True
        if answer in {"n", "no", ""}:
            return False
        if answer in {"a", "all"}:
            _approve_all = True
            return True
        print("Please enter y, n, or a.", flush=True)


def install_terminal_run_approval(monkeypatch, test_name: str):
    from interpreter.core.computer.terminal import terminal as terminal_mod

    original_run = terminal_mod.Terminal.run

    def approving_run(self, language, code, stream=False, display=False):
        if not prompt_for_code_execution(
            test_name=test_name, language=language, code=code
        ):
            pytest.fail(
                f"LLM-generated {language} code was not approved for {test_name!r}"
            )
        return original_run(self, language, code, stream=stream, display=display)

    monkeypatch.setattr(terminal_mod.Terminal, "run", approving_run)

