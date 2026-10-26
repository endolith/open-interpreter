from unittest import mock

import pytest
from rich.markdown import Markdown
from rich.padding import Padding

from interpreter.terminal_interface.utils import display_markdown_message as dmm


def test_display_markdown_message_renders_without_error(capsys):
    """display_markdown_message renders rules, tags, and plain markdown safely."""
    assert dmm.display_markdown_message("") is None
    assert dmm.display_markdown_message("---") is None
    assert dmm.display_markdown_message("> A status tag") is None
    assert dmm.display_markdown_message("Normal **bold** text") is None
    assert "bold" in capsys.readouterr().out


def test_renders_the_whole_message_as_one_markdown_block():
    """The full message is rendered as one Markdown object inside Padding.

    Rendering the whole document is what makes tables and multi-line constructs
    work; splitting on newlines would break them.
    """
    with mock.patch.object(dmm, "rich_print") as rp:
        dmm.display_markdown_message("alpha\nbeta")

    rp.assert_called_once()
    rendered = rp.call_args.args[0]
    assert isinstance(rendered, Padding)
    assert isinstance(rendered.renderable, Markdown)


def test_single_tag_line_gets_a_trailing_blank_line(capsys):
    """A lone ">"-prefixed line with no newline gets a blank line after it."""
    with mock.patch.object(dmm, "rich_print"):
        dmm.display_markdown_message("> only a tag")
    assert capsys.readouterr().out.endswith("\n")


def test_multiline_tag_does_not_get_an_extra_blank_line(capsys):
    """A multi-line message starting with ">" does not get the trailing blank line."""
    with mock.patch.object(dmm, "rich_print"):
        dmm.display_markdown_message("> tag\nsecond line")
    assert not capsys.readouterr().out.endswith("\n\n")


def test_unicode_encode_error_falls_back_to_plain_text(capsys):
    """A message Rich cannot encode is reported as plain text instead of raising."""
    with mock.patch.object(
        dmm, "rich_print", side_effect=UnicodeEncodeError("ascii", "x", 0, 1, "boom")
    ):
        assert dmm.display_markdown_message("unencodable line") is None
    assert "unencodable line" in capsys.readouterr().out


def test_surrogate_line_degrades_to_escaped_placeholder(capsys):
    """A line Rich cannot encode degrades to an escaped placeholder, not a crash."""
    with mock.patch.object(
        dmm, "rich_print", side_effect=UnicodeEncodeError("utf-8", "\ud800", 0, 1, "x")
    ):
        dmm.display_markdown_message("bad \ud800 line")
    out = capsys.readouterr().out
    assert "\\ud800" in out
    assert "\ud800" not in out


def test_non_unicode_errors_are_not_swallowed():
    """Only UnicodeEncodeError is caught; other exceptions keep propagating."""
    with mock.patch.object(dmm, "rich_print", side_effect=RuntimeError("unrelated")):
        with pytest.raises(RuntimeError):
            dmm.display_markdown_message("line")


def test_real_surrogate_line_does_not_raise():
    """End to end: a lone surrogate in the message never escapes display."""
    dmm.display_markdown_message("bad \ud800 line")
