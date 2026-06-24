from interpreter.terminal_interface.utils.export_to_markdown import (
    export_to_markdown,
    messages_to_markdown,
)


def test_user_message_gets_role_header():
    messages = [{"role": "user", "type": "message", "content": "Hello"}]
    md = messages_to_markdown(messages)
    assert "## user" in md
    assert "Hello" in md


def test_code_block_rendered():
    messages = [
        {"role": "assistant", "type": "code", "format": "python", "content": "1+1"}
    ]
    md = messages_to_markdown(messages)
    assert "```python" in md


def test_export_to_markdown_writes_file(tmp_path):
    path = tmp_path / "conversation.md"
    messages = [{"role": "user", "type": "message", "content": "Test"}]
    export_to_markdown(messages, str(path))
    assert path.read_text() == messages_to_markdown(messages)
