import os
import platform
from unittest import mock

import pytest
from _pytest.outcomes import Skipped

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


def test_require_bash_compatible_shell_rejects_fish(monkeypatch):
    """require_bash_compatible_shell() skips when SHELL points to fish on Unix."""
    if platform.system() == "Windows":
        pytest.skip("SHELL guard only applies to Unix")
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    with pytest.raises(Skipped, match="fish"):
        require_bash_compatible_shell()


def test_preprocess_shell_skips_active_line_prints_for_multiline():
    """Multiline code gets no active-line echoes (the marker would land wrong)."""
    code = "echo one \\\necho two"
    result = preprocess_shell(code)
    assert "##active_line" not in result
    assert "##end_of_execution##" in result


def test_preprocess_shell_adds_active_line_prints_for_single_line():
    """A single-line script gets active-line echoes when detection is on."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("INTERPRETER_ACTIVE_LINE_DETECTION", None)
        result = preprocess_shell("echo hi")

    assert '##active_line1##' in result
    assert "##end_of_execution##" in result


@pytest.mark.parametrize(
    "value",
    ["false", "False", "FALSE", "0", "no"],
)
def test_preprocess_shell_respects_active_line_detection_disabled(value):
    """Any value other than "true" (case-insensitively) disables active-line echoes."""
    with mock.patch.dict(os.environ, {"INTERPRETER_ACTIVE_LINE_DETECTION": value}):
        result = preprocess_shell("echo hi")
    assert "##active_line" not in result


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "TrUe"])
def test_preprocess_shell_respects_active_line_detection_enabled(value):
    """The env flag enables active-line echoes case-insensitively."""
    with mock.patch.dict(os.environ, {"INTERPRETER_ACTIVE_LINE_DETECTION": value}):
        result = preprocess_shell("echo hi")
    assert "##active_line1##" in result


def test_end_of_execution_marker_is_exact():
    """The appended sentinel is exactly the literal the detector matches.

    preprocess_shell appends the marker and detect_end_of_execution looks for it;
    the two must agree or the language would never report completion.
    """
    result = preprocess_shell("echo hi")
    assert result.endswith('echo "##end_of_execution##"')


@pytest.mark.parametrize(
    "snippet",
    [
        "echo one|",  # trailing pipe
        "echo one &&",
        "echo one ||",
        "if true",
        "while true",
        "for i in 1 2 3",
        "do",
        "then",
        "cat <(",
        "echo (",
        "if true; {",
    ],
)
def test_has_multiline_commands_detects_each_pattern(snippet):
    """Each documented continuation pattern is recognised as multiline."""
    assert has_multiline_commands(snippet)


def test_has_multiline_commands_false_for_plain_script():
    """A plain multi-statement script without continuations is not multiline."""
    assert not has_multiline_commands("echo one\necho two\nls -la")


def test_detect_active_line_parses_the_marker():
    """detect_active_line extracts the line number and returns None otherwise."""
    shell = Shell()
    assert shell.detect_active_line('echo "##active_line3##"') == 3
    assert shell.detect_active_line("no marker here") is None


def test_detect_end_of_execution_recognises_the_sentinel():
    """detect_end_of_execution is true only for the exact sentinel."""
    shell = Shell()
    assert shell.detect_end_of_execution('echo "##end_of_execution##"') is True
    assert shell.detect_end_of_execution("still running") is False


def test_shell_start_cmd_defaults_to_bash_when_shell_unset():
    """With no SHELL env var on Unix, Shell defaults to "bash"."""
    if platform.system() == "Windows":
        pytest.skip("SHELL default only applies to Unix")
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SHELL", None)
        shell = Shell()
    assert shell.start_cmd == ["bash"]
