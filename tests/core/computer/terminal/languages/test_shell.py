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


def test_detect_active_line_ignores_marker_without_a_line_number():
    """Shell.detect_active_line() returns None for '##active_line' with no digits.

    Program output is untrusted text that can contain the marker string (e.g.
    grepping these sources). Parsing it unconditionally raised ValueError,
    which killed the stdout reader thread so the end-of-execution marker was
    never seen and run() hung forever.
    """
    shell = Shell()
    assert shell.detect_active_line("##active_line\n") is None
    assert shell.detect_active_line("shell.py:31: '##active_line' in line\n") is None
    assert shell.detect_active_line("##active_line12##\n") == 12


@pytest.mark.linux_ci
@pytest.mark.timeout(30)
def test_shell_run_survives_active_line_text_in_output():
    """A shell block whose output contains '##active_line' still completes.

    The unparseable marker used to kill the stdout reader thread, hanging both
    this run() and every later one on the same Shell instance.
    """
    require_bash_compatible_shell()
    shell = Shell()
    try:
        output = "".join(
            chunk["content"]
            for chunk in shell.run("echo '##active_line'")
            if chunk.get("format") == "output"
        )
        assert "##active_line" in output
        # The instance must still be usable: the reader thread survived.
        again = "".join(
            chunk["content"]
            for chunk in shell.run("echo still_alive")
            if chunk.get("format") == "output"
        )
        assert "still_alive" in again
    finally:
        shell.terminate()


@pytest.mark.linux_ci
@pytest.mark.timeout(30)
def test_shell_run_returns_when_the_block_exits_the_shell():
    """A block that runs `exit` completes and reports that the shell died.

    `exit` (even a successful `exit 0`) kills bash before the appended
    ##end_of_execution## marker is reached, so run() waited for a marker that
    could never arrive and the turn never finished.
    """
    require_bash_compatible_shell()
    shell = Shell()
    try:
        output = "".join(
            chunk["content"]
            for chunk in shell.run("echo before\nexit 0\necho unreachable_line")
            if chunk.get("format") == "output"
        )
        assert "before" in output
        assert "exited with code 0" in output
        assert "unreachable_line" not in output
        assert shell.process is None
    finally:
        shell.terminate()


def test_require_bash_compatible_shell_rejects_fish(monkeypatch):
    """require_bash_compatible_shell() fails when SHELL points to fish on Unix."""
    if platform.system() == "Windows":
        pytest.skip("SHELL guard only applies to Unix")
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    with pytest.raises(Failed, match="fish"):
        require_bash_compatible_shell()
