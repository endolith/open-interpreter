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
