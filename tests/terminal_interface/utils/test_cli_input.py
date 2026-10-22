"""Multi-line input handling for pasted blocks.

cli_input requests GNU Readline's bracketed-paste setting. On libedit
(often the macOS backend) that request is a no-op; preserving pasted tabs
there is tracked in #535. The one-line early return is backend-independent.
"""

from unittest import mock

import pytest

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


def test_single_line_triple_quoted_paste_returns_immediately(monkeypatch):
    """Both markers on one input line must not wait for another line."""
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        if len(prompts) > 1:
            raise AssertionError("cli_input waited for a second line")
        return '"""hello"""'

    monkeypatch.setattr("builtins.input", fake_input)

    assert cli_input() == '"""hello"""'
    assert len(prompts) == 1


def test_requests_bracketed_paste(monkeypatch):
    """cli_input asks readline for bracketed paste; the effect is backend-specific."""
    readline = pytest.importorskip("readline")
    calls = []
    monkeypatch.setattr(readline, "parse_and_bind", lambda spec: calls.append(spec))
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")

    cli_input()

    assert "set enable-bracketed-paste on" in calls


def test_missing_readline_is_ignored(monkeypatch):
    """Platforms without readline still read input."""
    import sys

    monkeypatch.setitem(sys.modules, "readline", None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")

    assert cli_input() == "hello"
