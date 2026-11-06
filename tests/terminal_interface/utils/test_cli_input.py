"""Multi-line input handling for pasted blocks.

cli_input requests GNU Readline's bracketed-paste setting. On libedit
(often the macOS backend) that request is a no-op; preserving pasted tabs
there is tracked in #535. The one-line early return is backend-independent.
"""

from interpreter.core.utils.prompt_choice import prompt_choice
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


def test_cli_input_strips_mouse_reports(monkeypatch):
    """Mouse-report bytes arriving mid-prompt never reach the caller."""
    monkeypatch.setattr(
        "builtins.input", lambda _prompt="": "hi\x1b[<35;25;57M\x1b[61;4R"
    )
    assert cli_input("> ") == "hi"


def test_cli_input_mouse_only_becomes_empty(monkeypatch):
    """A prompt holding only leaked reports reads as blank (ignored downstream)."""
    monkeypatch.setattr(
        "builtins.input", lambda _prompt="": "\x1b[<35;25;57M\x1b[61;4R"
    )
    assert cli_input("> ") == ""


def test_prompt_choice_ignores_leaked_report_bytes(monkeypatch):
    """Leaked tracking bytes around a y/n answer don't fail validation."""
    monkeypatch.setattr(
        "builtins.input", lambda _prompt="": "\x1b[<0;80;53mY\x1b[61;4R"
    )
    assert prompt_choice("  ", ("y", "n")) == "y"


def test_prompt_choice_rejects_spelled_out_answer(monkeypatch):
    """"yes" is not an answer: it re-prompts instead of counting as "y".

    Truncating to the first letter would let a multi-key or pasted answer act
    as a deliberate choice, so the prompt demands a single keypress.
    """
    answers = iter(["yes", "y"])
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)
    assert prompt_choice("  Retry? (y/n) ", ("y", "n")) == "y"
    # The full prompt is shown once, then only the minimal reprompt.
    assert prompts == ["  Retry? (y/n) ", "  "]


def test_prompt_choice_rejects_multi_letter_run(monkeypatch):
    """A run of valid letters ("yqn") is not a choice either.

    Every letter is one the user could have meant, so reading the first one
    would guess on their behalf; the answer has to be unambiguous.
    """
    answers = iter(["yqn", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert prompt_choice("  ", ("y", "a", "n")) == "n"


def test_prompt_choice_rejects_blank_line(monkeypatch):
    """An empty line re-prompts rather than resolving to a choice."""
    answers = iter(["", "  ", "a"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert prompt_choice("  ", ("y", "a", "n")) == "a"


def test_prompt_choice_tolerates_case_and_padding(monkeypatch):
    """A single letter still answers regardless of case or stray spaces."""
    monkeypatch.setattr("builtins.input", lambda _prompt="": "  Y  ")
    assert prompt_choice("  ", ("y", "n")) == "y"


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
