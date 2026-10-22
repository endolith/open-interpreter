"""Multi-line input must not hang on a paste that is already complete.

With bracketed paste the terminal delivers a multi-line paste (embedded
newlines) in a single ``input()`` call. The old loop always waited for a
*second* line containing the closing marker, so a one-line
``\"\"\"...\"\"\"`` paste hung forever. The fix also enables bracketed paste so
pasted tabs survive readline's Tab-completion binding.
"""

from unittest import mock

import pytest

from interpreter.terminal_interface.utils.cli_input import cli_input


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


def test_multiline_input_still_collects_lines(monkeypatch):
    """The old line-by-line mode keeps working when the block is split."""
    responses = iter(['"""hello', "world", '"""'])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    assert cli_input() == '"""hello\nworld\n"""'


def test_single_line_input_is_unchanged(monkeypatch):
    """Ordinary input is returned as-is."""
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")

    assert cli_input() == "hello"


def test_enables_bracketed_paste(monkeypatch):
    """readline is configured to deliver pastes literally (tabs preserved)."""
    readline = pytest.importorskip("readline")
    calls = []
    monkeypatch.setattr(readline, "parse_and_bind", lambda spec: calls.append(spec))
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")

    cli_input()

    assert "set enable-bracketed-paste on" in calls


def test_missing_readline_is_ignored(monkeypatch):
    """Platforms without readline (e.g. Windows) still read input."""
    import sys

    monkeypatch.setitem(sys.modules, "readline", None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")

    assert cli_input() == "hello"
