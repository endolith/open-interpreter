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
