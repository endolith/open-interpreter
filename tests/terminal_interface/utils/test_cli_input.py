"""Multi-line input handling for pasted blocks.

cli_input preserves pasted tabs on both readline backends: GNU Readline
enables bracketed paste, while libedit (often the macOS backend) temporarily
binds Tab to literal insertion and restores completion afterwards (#535). The
one-line early return is backend-independent.
"""

from unittest import mock

import pytest

from interpreter.terminal_interface.utils.cli_input import _readline_backend, cli_input


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


def test_gnu_backend_requests_bracketed_paste(monkeypatch):
    """GNU Readline gets bracketed paste and no Tab rebinding.

    Pinned to the GNU backend because CI runs macOS, where libedit is the
    native readline and correctly takes the other branch.
    """
    readline = pytest.importorskip("readline")
    calls = []
    monkeypatch.setattr(readline, "parse_and_bind", lambda spec: calls.append(spec))
    monkeypatch.setattr(readline, "backend", "gnu", raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")

    cli_input()

    assert calls == ["set enable-bracketed-paste on"]


def test_libedit_binds_tab_to_literal_insert(monkeypatch):
    """libedit gets Tab bound to literal insertion so pasted tabs survive.

    libedit treats Tab as a completion request, which swallows tabs pasted from
    spreadsheet TSV data (#535).
    """
    readline = pytest.importorskip("readline")
    calls = []
    monkeypatch.setattr(readline, "parse_and_bind", lambda spec: calls.append(spec))
    monkeypatch.setattr(readline, "backend", "libedit", raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "a\tb")

    assert cli_input() == "a\tb"
    assert calls == ["bind ^I ed-insert", "bind ^I rl_complete"]


def test_libedit_restores_tab_completion_after_multiline_input(monkeypatch):
    """The Tab binding is undone after a multi-line read too.

    Restoration must happen once the whole block has been collected, so pasted
    tabs on continuation lines are preserved as well.
    """
    readline = pytest.importorskip("readline")
    calls = []
    monkeypatch.setattr(readline, "parse_and_bind", lambda spec: calls.append(spec))
    monkeypatch.setattr(readline, "backend", "libedit", raising=False)
    lines = ['start """', "a\tb", 'end """']
    monkeypatch.setattr("builtins.input", lambda prompt="": lines.pop(0))

    assert cli_input() == 'start """\na\tb\nend """'
    assert calls == ["bind ^I ed-insert", "bind ^I rl_complete"]


def test_libedit_restores_tab_completion_after_interrupt(monkeypatch):
    """Ctrl-C during the prompt still restores Tab completion.

    Otherwise an interrupted prompt would leave the shell with Tab bound to
    literal insertion, breaking completion for the rest of the session.
    """
    readline = pytest.importorskip("readline")
    calls = []
    monkeypatch.setattr(readline, "parse_and_bind", lambda spec: calls.append(spec))
    monkeypatch.setattr(readline, "backend", "libedit", raising=False)

    def interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)

    with pytest.raises(KeyboardInterrupt):
        cli_input()

    assert calls == ["bind ^I ed-insert", "bind ^I rl_complete"]


def test_readline_backend_prefers_backend_attribute():
    """Python 3.13+'s readline.backend attribute is used when present."""

    class FakeReadline:
        backend = "libedit"
        __doc__ = "GNU Readline"  # must be ignored when backend is authoritative

    assert _readline_backend(FakeReadline) == "libedit"


def test_readline_backend_falls_back_to_docstring():
    """Older builds, which lack readline.backend, are identified by docstring.

    readline 3.11/3.12 exposes the implementation only in the module doc,
    whose libedit build mentions "libedit".
    """

    class FakeReadline:
        __doc__ = "This module provides readline functionality using libedit."

    assert _readline_backend(FakeReadline) == "libedit"


def test_readline_backend_defaults_to_gnu():
    """An unreadable backend is assumed to be GNU Readline.

    GNU is the branching default on Linux/Windows; guessing libedit there would
    needlessly rebind Tab.
    """

    class FakeReadline:
        __doc__ = None

    assert _readline_backend(FakeReadline) == "gnu"


def test_missing_readline_is_ignored(monkeypatch):
    """Platforms without readline still read input."""
    import sys

    monkeypatch.setitem(sys.modules, "readline", None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")

    assert cli_input() == "hello"
