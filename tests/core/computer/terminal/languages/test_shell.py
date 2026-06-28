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
    code = "echo one\necho two"
    result = add_active_line_prints(code)
    assert 'echo "##active_line1##"' in result


def test_preprocess_shell_adds_end_marker():
    result = preprocess_shell("echo hi")
    assert "##end_of_execution##" in result


def test_has_multiline_commands_detects_line_continuation():
    assert has_multiline_commands("echo hello \\\nworld")


def test_shell_start_cmd_uses_shell_env():
    """Shell subprocess uses os.environ['SHELL'] on Unix; cmd.exe on Windows."""
    import os

    if platform.system() == "Windows":
        shell = Shell()
        assert shell.start_cmd == ["cmd.exe"]
    else:
        with mock.patch.dict(os.environ, {"SHELL": "/bin/bash"}):
            shell = Shell()
        assert shell.start_cmd == ["/bin/bash"]


def test_require_bash_compatible_shell_rejects_fish(monkeypatch):
    if platform.system() == "Windows":
        pytest.skip("SHELL guard only applies to Unix")
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    with pytest.raises(Failed, match="fish"):
        require_bash_compatible_shell()
