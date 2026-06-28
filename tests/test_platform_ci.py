"""Minimal per-OS CI smokes: real subprocess / path behavior not covered by mocks.

Linux CI runs the full unit suite (see test_language_subprocess.py for language
e2e). Windows and macOS CI jobs run only tests marked ``windows_ci`` or
``darwin_ci`` here — cmd.exe, PowerShell, AppleScript, and related quirks from
cross-platform test development.
"""

import os

import pytest

from interpreter import OpenInterpreter
from interpreter.core.computer.terminal.languages.shell import Shell
from tests.helpers import console_output_text


@pytest.fixture
def interpreter():
    oi = OpenInterpreter()
    oi.conversation_history = False
    return oi


# --- Windows (cmd.exe, PowerShell, USERPROFILE paths, test package imports) ---


@pytest.mark.windows_ci
def test_windows_imports_tests_package():
    """Regression: ``from tests.helpers import ...`` must work (pyproject pythonpath)."""

    from tests.helpers import TEST_LLM_MODEL

    assert TEST_LLM_MODEL


@pytest.mark.windows_ci
def test_shell_start_cmd_uses_cmd_exe():
    shell = Shell()
    assert shell.start_cmd == ["cmd.exe"]


@pytest.mark.windows_ci
@pytest.mark.subprocess_e2e
@pytest.mark.timeout(30)
def test_shell_cmd_echo_smoke(interpreter):
    chunks = list(interpreter.computer.run("shell", "echo shell_ok"))
    assert "shell_ok" in console_output_text(chunks)


@pytest.mark.windows_ci
@pytest.mark.subprocess_e2e
@pytest.mark.timeout(30)
def test_shell_cmd_nested_loop_quoting(interpreter):
    """cmd.exe nested ``for`` loops — distinct quoting from bash (Linux CI)."""

    code = "for %i in (a b) do for %j in (1 2) do echo %i_%j"
    chunks = list(interpreter.computer.run("shell", code))
    output = console_output_text(chunks)
    assert "a_1" in output
    assert "b_2" in output


@pytest.mark.windows_ci
@pytest.mark.subprocess_e2e
@pytest.mark.timeout(30)
def test_powershell_subprocess_smoke(interpreter):
    chunks = list(interpreter.computer.run("powershell", 'Write-Output "ps_ok"'))
    assert "ps_ok" in console_output_text(chunks)


# --- macOS (osascript, Unix $SHELL) ---


@pytest.mark.darwin_ci
def test_shell_start_cmd_uses_shell_env():
    shell = Shell()
    assert shell.start_cmd == [os.environ.get("SHELL", "bash")]


@pytest.mark.darwin_ci
@pytest.mark.subprocess_e2e
@pytest.mark.timeout(30)
def test_shell_bash_nested_loop_quoting(interpreter):
    code = 'for i in a b; do for j in 1 2; do echo "${i}_${j}"; done; done'
    chunks = list(interpreter.computer.run("shell", code))
    output = console_output_text(chunks)
    assert "a_1" in output
    assert "b_2" in output


@pytest.mark.darwin_ci
@pytest.mark.subprocess_e2e
@pytest.mark.timeout(30)
def test_applescript_subprocess_smoke(interpreter):
    chunks = list(interpreter.computer.run("applescript", 'return "as_ok"'))
    assert "as_ok" in console_output_text(chunks)
