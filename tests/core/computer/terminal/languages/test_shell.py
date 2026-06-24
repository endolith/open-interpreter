from interpreter.core.computer.terminal.languages.shell import (
    Shell,
    add_active_line_prints,
    has_multiline_commands,
    preprocess_shell,
)


def test_add_active_line_prints():
    code = "echo one\necho two"
    result = add_active_line_prints(code)
    assert 'echo "##active_line1##"' in result


def test_preprocess_shell_adds_end_marker():
    result = preprocess_shell("echo hi")
    assert "##end_of_execution##" in result


def test_has_multiline_commands_detects_line_continuation():
    assert has_multiline_commands("echo hello \\\nworld")


def test_shell_detect_active_line():
    shell = Shell()
    assert shell.detect_active_line('echo "##active_line3##"') == 3
