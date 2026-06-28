import os
from unittest import mock

import pytest

from tests import integration_support as support


def test_integration_tests_allowed_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OI_RUN_INTEGRATION", "1")
    assert support.integration_tests_allowed() is False


def test_integration_tests_allowed_requires_local_opt_in(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("OI_RUN_INTEGRATION", raising=False)
    assert support.integration_tests_allowed() is False


def test_integration_tests_allowed_on_ci(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("OI_RUN_INTEGRATION", raising=False)
    assert support.integration_tests_allowed() is True


def test_prompt_for_code_execution_can_approve_all(monkeypatch):
    support._approve_all = False
    support._approved_hashes.clear()
    monkeypatch.setattr(support.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "a")
    assert support.prompt_for_code_execution(
        test_name="demo", language="python", code='print("hi")'
    )
    assert support.prompt_for_code_execution(
        test_name="demo", language="python", code='print("other")'
    )


def test_install_terminal_run_approval_blocks_when_denied(monkeypatch):
    support._approve_all = False
    support._approved_hashes.clear()
    monkeypatch.setattr(support, "prompt_for_code_execution", lambda **_: False)

    from interpreter.core.computer.terminal.terminal import Terminal

    install = support.install_terminal_run_approval
    install(monkeypatch, "demo_test")

    terminal = Terminal(computer=mock.Mock())
    with pytest.raises(pytest.fail.Exception, match="not approved"):
        terminal.run("python", 'print("hi")')
