import platform
from unittest import mock

import pytest
from _pytest.outcomes import Failed

from interpreter.core.computer.terminal.languages.shell import (
    Shell,
    add_active_line_prints,
    has_multiline_commands,
    preprocess_shell,
)
from tests.helpers import require_bash_compatible_shell


def test_add_active_line_prints():
    """add_active_line_prints() prefixes shell commands with ##active_lineN## echo markers."""
    code = "echo one\necho two"
    result = add_active_line_prints(code)
    assert 'echo "##active_line1##"' in result


def test_preprocess_shell_adds_end_marker():
    """preprocess_shell() appends an ##end_of_execution## marker to shell code."""
    result = preprocess_shell("echo hi")
    assert "##end_of_execution##" in result


def test_has_multiline_commands_detects_line_continuation():
    """has_multiline_commands() detects backslash line continuations in shell scripts."""
    assert has_multiline_commands("echo hello \\\nworld")


def test_shell_start_cmd_uses_bash_on_unix():
    """On Unix, Shell always invokes bash (not $SHELL) because emitted code is bash syntax."""
    if platform.system() == "Windows":
        pytest.skip("Windows uses cmd.exe")
    shell = Shell()
    assert shell.start_cmd == ["bash", "--norc", "--noprofile"]


def test_shell_start_cmd_uses_cmd_on_windows():
    """On Windows, Shell invokes cmd.exe."""
    if platform.system() != "Windows":
        pytest.skip("cmd.exe path only applies on Windows")
    shell = Shell()
    assert shell.start_cmd == ["cmd.exe"]


def test_require_bash_compatible_shell_rejects_fish(monkeypatch):
    """require_bash_compatible_shell() fails when SHELL points to fish on Unix."""
    if platform.system() == "Windows":
        pytest.skip("SHELL guard only applies to Unix")
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    with pytest.raises(Failed, match="fish"):
        require_bash_compatible_shell()
