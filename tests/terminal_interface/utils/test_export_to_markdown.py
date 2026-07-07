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
