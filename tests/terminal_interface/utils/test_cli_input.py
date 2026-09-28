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


def test_prompt_is_forwarded_to_input():
    """cli_input passes its prompt argument straight to input().

    Callers supply the prompt text; dropping or wrapping it would show the
    wrong prompt while still reading the right value.
    """
    with mock.patch("builtins.input", return_value="ok") as inp:
        cli_input("Enter something: ")

    assert inp.call_args.args == ("Enter something: ",)


def test_default_prompt_is_empty_string():
    """Omitting the prompt calls input("") rather than input(None).

    input(None) prints "None" as the prompt on some platforms, so the default
    must be the empty string.
    """
    with mock.patch("builtins.input", return_value="ok") as inp:
        cli_input()

    assert inp.call_args.args == ("",)
