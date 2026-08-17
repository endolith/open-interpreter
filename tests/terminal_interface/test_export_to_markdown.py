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


def test_include_reasoning_false_omits_reasoning_blocks():
    """With include_reasoning=False (the %markdown_final path), reasoning
    messages must be dropped entirely from the export so only the final answers
    and code remain. The reason block is stripped along with its content, and
    the assistant response that follows still gets its own header."""
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
    markdown = messages_to_markdown(messages, include_reasoning=False)
    assert "The user is asking" not in markdown
    assert "check a forecast" not in markdown
    assert "> The user is asking" not in markdown
    assert "Checking now." in markdown


def test_include_reasoning_false_keeps_assistant_header():
    """When reasoning is omitted, the header for the assistant section must be
    emitted by the real response that follows, not swallowed by the skipped
    reasoning block. This prevents an answer from appearing without its
    '## assistant' header."""
    messages = [
        {"role": "user", "type": "message", "content": "Hi"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "Thinking..."},
        {"role": "assistant", "type": "message", "content": "The answer."},
    ]
    markdown = messages_to_markdown(messages, include_reasoning=False)
    assert "## assistant\n\nThe answer." in markdown


def test_include_reasoning_true_default_keeps_reasoning():
    """The default (include_reasoning=True) must keep exporting reasoning as
    blockquotes, matching %markdown behavior; the option only kicks in when
    explicitly disabled."""
    messages = [
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "Thought."},
        {"role": "assistant", "type": "message", "content": "Answer."},
    ]
    markdown = messages_to_markdown(messages)
    assert "> Thought." in markdown
