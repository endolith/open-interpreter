from unittest import mock

from interpreter.terminal_interface.utils.cli_input import cli_input


def test_single_line_input():
    """cli_input returns a single line when the user does not start a multiline block."""
    with mock.patch("builtins.input", return_value="hello"):
        assert cli_input("> ") == "hello"


def test_multiline_input():
    """cli_input collects lines until a closing triple-quote delimiter is entered."""
    lines = ['start """', "line one", "line two", 'end """']
    with mock.patch("builtins.input", side_effect=lines):
        result = cli_input()
    assert result == 'start """\nline one\nline two\nend """'
