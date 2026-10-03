from interpreter.terminal_interface.utils.export_to_markdown import (
    export_to_markdown,
    messages_to_markdown,
)


def test_user_message_gets_role_header():
    """User messages are rendered with a ## role header in markdown."""
    messages = [{"role": "user", "type": "message", "content": "Hello"}]
    md = messages_to_markdown(messages)
    assert "## user" in md
    assert "Hello" in md


def test_consecutive_user_messages_each_get_header():
    """Two user messages in a row each get their own ## user section in the export."""
    messages = [
        {"role": "user", "type": "message", "content": "First"},
        {"role": "user", "type": "message", "content": "Second"},
    ]
    md = messages_to_markdown(messages)
    assert md == "## user\n\nFirst\n\n## user\n\nSecond\n\n"


def test_code_block_rendered():
    """Assistant code messages are wrapped in a fenced code block with the format language."""
    messages = [
        {"role": "assistant", "type": "code", "format": "python", "content": "1+1"}
    ]
    md = messages_to_markdown(messages)
    assert "```python" in md


def test_export_to_markdown_writes_file(tmp_path):
    """export_to_markdown writes the rendered markdown to the given file path."""
    path = tmp_path / "conversation.md"
    messages = [{"role": "user", "type": "message", "content": "Test"}]
    export_to_markdown(messages, str(path))
    assert path.read_text() == messages_to_markdown(messages)


def test_empty_messages_returns_empty_string():
    """An empty conversation exports to an empty markdown string."""
    assert messages_to_markdown([]) == ""


def test_console_block_rendered():
    """Console output messages are wrapped in a fenced block using their format."""
    messages = [
        {
            "role": "assistant",
            "type": "console",
            "format": "output",
            "content": "printed",
        }
    ]
    md = messages_to_markdown(messages)
    assert md == "## assistant\n\n```output\nprinted\n```\n\n"


def test_consecutive_assistant_messages_share_one_header():
    """Two messages from the same non-user role render under a single header.

    The header is emitted only when the role changes; if the previous-role
    bookkeeping broke, every chunk would repeat "## assistant", bloating the
    export.
    """
    messages = [
        {"role": "assistant", "type": "message", "content": "one"},
        {"role": "assistant", "type": "message", "content": "two"},
    ]

    md = messages_to_markdown(messages)

    assert md == "## assistant\n\none\n\ntwo\n\n"
    assert md.count("## assistant") == 1


def test_role_change_starts_a_new_header():
    """A change of role emits a fresh header for the new role.

    previous_role must be updated when the header is written, or the second
    role would be appended under the first role's header.
    """
    messages = [
        {"role": "assistant", "type": "message", "content": "a"},
        {"role": "computer", "type": "message", "content": "b"},
    ]

    md = messages_to_markdown(messages)

    assert md == "## assistant\n\na\n\n## computer\n\nb\n\n"


def test_message_type_is_matched_exactly():
    """Only chunks whose type is exactly "message" get their content appended.

    The literal is compared, so a renamed or case-changed type would drop the
    content from the export even though the chunk is a message.
    """
    messages = [
        {"role": "assistant", "type": "message", "content": "kept"},
        {"role": "assistant", "type": "status", "content": "dropped"},
    ]

    md = messages_to_markdown(messages)

    assert "kept" in md
    assert "dropped" not in md


def test_message_content_is_followed_by_a_blank_line():
    """A message body is joined with exactly one trailing blank line.

    The separator is "\n\n"; a wrapped or shifted one would change the rendered
    spacing (and the exact-equality tests above would catch it, pinned here as
    the reason).
    """
    messages = [{"role": "assistant", "type": "message", "content": "x"}]

    assert messages_to_markdown(messages) == "## assistant\n\nx\n\n"

