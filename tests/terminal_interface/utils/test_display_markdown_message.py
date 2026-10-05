import pytest
from interpreter.terminal_interface.utils.display_markdown_message import (
    display_markdown_message,
)


def test_display_markdown_message_renders_without_error(capsys):
    """display_markdown_message renders rules, tags, and plain markdown safely."""
    assert display_markdown_message("") is None
    assert display_markdown_message("---") is None
    assert display_markdown_message("> A status tag") is None
    assert display_markdown_message("Normal **bold** text") is None

    # At least the non-empty messages should have produced some output.
    assert "bold" in capsys.readouterr().out


from unittest import mock

from interpreter.terminal_interface.utils import display_markdown_message as dmm


def test_lines_are_split_on_newlines_and_markdown_rendered():
    """Each non-empty line is rendered as its own Markdown object.

    The split is on "\\n"; a mutated delimiter would pass the whole message as
    one line and lose the per-line rendering.
    """
    with mock.patch.object(dmm, "rich_print") as rp:
        dmm.display_markdown_message("alpha\nbeta")

    rendered = [c.args[0] for c in rp.call_args_list]
    assert len(rendered) == 2
    assert all(isinstance(obj, dmm.Markdown) for obj in rendered)


def test_rule_line_renders_a_white_rule():
    """A line of exactly "---" becomes a white Rule.

    The comparison is exact and the style literal is "white"; a changed literal
    would change the rule color.
    """
    with mock.patch.object(dmm, "rich_print") as rp:
        dmm.display_markdown_message("---")

    rule = rp.call_args.args[0]
    assert isinstance(rule, dmm.Rule)
    assert rule.style == "white"


def test_blank_lines_print_empty(capsys):
    """Blank lines are printed as empty lines, not rendered as markdown.

    Asserts the exact line structure rather than a newline count: two non-empty
    lines already produce two newlines, so a count check would still pass if the
    blank line between them were dropped or rendered as markdown. Lines are
    stripped because Rich pads rendered output to the console width, which says
    nothing about the content.
    """
    dmm.display_markdown_message("a\n\nb")
    lines = capsys.readouterr().out.splitlines()
    assert [line.strip() for line in lines] == ["a", "", "b"]


def test_single_tag_line_gets_a_trailing_blank_line(capsys):
    """A lone ">"-prefixed line with no newline gets a blank line after it.

    The trailing print is an aesthetic choice for tags; the condition checks
    both the absence of a newline and the leading ">".
    """
    with mock.patch.object(dmm, "rich_print"):
        dmm.display_markdown_message("> only a tag")
    # The blank line from the final print is the last output.
    assert capsys.readouterr().out.endswith("\n")


def test_multiline_tag_does_not_get_an_extra_blank_line(capsys):
    """A multi-line message starting with ">" does not get the trailing blank line.

    The condition requires "\n" not in message; a message with newlines must not
    take the extra print.
    """
    with mock.patch.object(dmm, "rich_print"):
        dmm.display_markdown_message("> tag\nsecond line")
    out = capsys.readouterr().out
    assert not out.endswith("\n\n")


def test_single_plain_line_gets_no_trailing_blank_line(capsys):
    """A single-line message that is not a ">" tag gets no extra blank line.

    The final condition requires message.startswith(">"); with `or` a plain
    single-line message would also get the extra print.
    """
    with mock.patch.object(dmm, "rich_print"):
        dmm.display_markdown_message("plain line")

    assert not capsys.readouterr().out.endswith("\n")


def test_unicode_encode_error_falls_back_to_a_plain_line(capsys):
    """A line Rich cannot encode is reported as plain text instead of raising.

    The except branch is the whole reason the try exists: a stray character in
    model output would otherwise propagate out of the display path and abort the
    turn that was printing it.
    """
    with mock.patch.object(
        dmm, "rich_print", side_effect=UnicodeEncodeError("ascii", "x", 0, 1, "boom")
    ):
        assert dmm.display_markdown_message("unencodable line") is None

    out = capsys.readouterr().out
    assert "Error displaying line: unencodable line" in out


def test_non_unicode_errors_are_not_swallowed():
    """Only UnicodeEncodeError is caught; other exceptions keep propagating.

    The except clause names one exception type. Widening it to a bare except
    would silently hide genuine bugs in Markdown construction, so this pins the
    narrowness from the other side.
    """
    with mock.patch.object(dmm, "rich_print", side_effect=RuntimeError("unrelated")):
        with pytest.raises(RuntimeError):
            dmm.display_markdown_message("line")


def test_a_failing_line_does_not_stop_later_lines(capsys):
    """One unencodable line does not prevent the rest of the message rendering.

    The handler continues the loop rather than returning, so a single bad
    character cannot swallow every subsequent line of output.
    """
    calls = []

    def flaky(obj):
        calls.append(obj)
        if len(calls) == 1:
            raise UnicodeEncodeError("ascii", "x", 0, 1, "boom")

    with mock.patch.object(dmm, "rich_print", side_effect=flaky):
        dmm.display_markdown_message("first\nsecond")

    assert len(calls) == 2
    assert "Error displaying line:" in capsys.readouterr().out


def test_the_trailing_print_is_an_empty_string_not_a_placeholder():
    """The tag line is padded with a genuinely empty line.

    Mocking rich_print means no other output is produced, so a placeholder like
    "None" or any text at all would look identical to a blank line if the test
    only checked that output ends in a newline. Capturing the print arguments
    pins the actual value.
    """
    printed = []
    with mock.patch.object(dmm, "rich_print"):
        with mock.patch("builtins.print", side_effect=lambda *a: printed.append(a)):
            dmm.display_markdown_message("> only a tag")

    assert printed == [("",)]


def test_a_multiline_tag_prints_no_trailing_line_at_all():
    """With newlines present, the trailing padding is skipped entirely.

    The condition tests for the absence of a newline, so the check cannot be a
    trailing-newline comparison once rich_print is mocked — that yields the same
    output whether the final print runs or not. Counting the calls is the only
    way to see the difference.
    """
    printed = []
    with mock.patch.object(dmm, "rich_print"):
        with mock.patch("builtins.print", side_effect=lambda *a: printed.append(a)):
            dmm.display_markdown_message("> tag\nsecond line")

    assert printed == []
