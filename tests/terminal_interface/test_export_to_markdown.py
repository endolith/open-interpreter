from interpreter.terminal_interface.utils.export_to_markdown import (
    messages_to_markdown,
)


def test_reasoning_message_exported_as_blockquote():
    """Reasoning messages (format="reasoning") must be exported as a markdown
    blockquote so the model's thoughts are visually distinct from its actual
    response, matching how they are displayed in the terminal as a "Thinking"
    panel. Without this, the thinking is dumped as plain text and reads like a
    normal assistant answer."""
    messages = [
        {"role": "user", "type": "message", "content": "What's the weather?"},
        {
            "role": "assistant",
            "type": "message",
            "format": "reasoning",
            "content": "The user is asking about the weather.\nI should check a forecast.",
        },
        {"role": "assistant", "type": "message", "content": "Checking now."},
    ]
    markdown = messages_to_markdown(messages)
    assert "> The user is asking about the weather.\n> I should check a forecast." in markdown
    assert "Checking now." in markdown
    assert "> Checking now." not in markdown


def test_reasoning_message_export_handles_trailing_newlines():
    """Reasoning chunks are often yielded with trailing newlines already
    attached; the exporter must strip them so the blockquote does not end with a
    stray blank quoted line or extra spacing before the next section."""
    messages = [
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "Thought.\n\n\n"},
        {"role": "assistant", "type": "message", "content": "Answer."},
    ]
    markdown = messages_to_markdown(messages)
    assert "> Thought.\n\nAnswer." in markdown


def test_normal_message_unchanged():
    """Ordinary (non-reasoning) messages must keep exporting as plain markdown;
    only reasoning messages get blockquoted."""
    messages = [
        {"role": "user", "type": "message", "content": "Hi"},
        {"role": "assistant", "type": "message", "content": "Hello!"},
    ]
    markdown = messages_to_markdown(messages)
    assert "## user\n\nHi" in markdown
    assert "## assistant\n\nHello!" in markdown
