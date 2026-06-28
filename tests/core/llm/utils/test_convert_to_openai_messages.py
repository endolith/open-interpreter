import json
from types import SimpleNamespace

import pytest

from interpreter.core.llm.utils.convert_to_openai_messages import convert_to_openai_messages


@pytest.fixture
def interpreter():
    return SimpleNamespace(
        user_message_template="User: {content}",
        always_apply_user_message_template=False,
        code_output_template="Output:\n{content}",
        empty_code_output_template="(no output)",
        code_output_sender="user",
        debug=False,
    )


def test_assistant_message_converted(interpreter):
    messages = [{"role": "assistant", "type": "message", "content": "Hello"}]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "assistant", "content": "Hello"}]


def test_last_user_message_gets_template(interpreter):
    messages = [{"role": "user", "type": "message", "content": "Hello"}]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "user", "content": "User: Hello"}]


def test_code_with_function_calling(interpreter):
    messages = [
        {"role": "assistant", "type": "code", "format": "python", "content": "1+1"}
    ]
    result = convert_to_openai_messages(
        messages, function_calling=True, interpreter=interpreter
    )
    assert result[0]["role"] == "assistant"
    assert result[0]["function_call"]["name"] == "execute"
    args = json.loads(result[0]["function_call"]["arguments"])
    assert args == {"language": "python", "code": "1+1"}


def test_code_without_function_calling(interpreter):
    messages = [
        {"role": "assistant", "type": "code", "format": "python", "content": "1+1"}
    ]
    result = convert_to_openai_messages(
        messages, function_calling=False, interpreter=interpreter
    )
    assert result[0]["content"] == "```python\n1+1\n```"


def test_console_output_function_role(interpreter):
    messages = [
        {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": "42",
        }
    ]
    result = convert_to_openai_messages(
        messages, function_calling=True, interpreter=interpreter
    )
    assert result == [{"role": "function", "name": "execute", "content": "42"}]


def test_console_empty_output(interpreter):
    messages = [
        {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": "   ",
        }
    ]
    result = convert_to_openai_messages(
        messages, function_calling=True, interpreter=interpreter
    )
    assert result[0]["content"] == "No output"


def test_recipient_not_assistant_skipped(interpreter):
    messages = [
        {
            "role": "user",
            "type": "message",
            "content": "hidden",
            "recipient": "user",
        }
    ]
    assert convert_to_openai_messages(messages, interpreter=interpreter) == []


def test_image_description_passes_through(interpreter):
    messages = [
        {
            "role": "user",
            "type": "image",
            "format": "description",
            "content": "A gradient image",
        }
    ]
    result = convert_to_openai_messages(messages, vision=False, interpreter=interpreter)
    assert result == [{"role": "user", "content": "A gradient image"}]


def test_file_message(interpreter):
    messages = [{"role": "user", "type": "file", "content": "/path/to/file.txt"}]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "user", "content": "/path/to/file.txt"}]


def test_unknown_type_raises(interpreter):
    with pytest.raises(Exception, match="Unable to convert"):
        convert_to_openai_messages(
            [{"role": "user", "type": "unknown", "content": "x"}],
            interpreter=interpreter,
        )


def test_console_output_assistant_sender(interpreter):
    interpreter.code_output_sender = "assistant"
    messages = [
        {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": "result",
        }
    ]
    result = convert_to_openai_messages(
        messages, function_calling=False, interpreter=interpreter
    )
    assert result[0]["role"] == "assistant"
    assert "result" in result[0]["content"]


def test_vision_false_skips_base64_image(interpreter):
    import base64

    png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    messages = [
        {
            "role": "user",
            "type": "image",
            "format": "base64.png",
            "content": png,
        }
    ]
    assert (
        convert_to_openai_messages(messages, vision=False, interpreter=interpreter) == []
    )


def test_image_path_with_vision(tmp_path, interpreter):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01")
    messages = [
        {
            "role": "user",
            "type": "image",
            "format": "path",
            "content": str(img),
        }
    ]
    result = convert_to_openai_messages(
        messages, vision=True, shrink_images=False, interpreter=interpreter
    )
    assert result[0]["content"][0]["type"] == "image_url"
    text_parts = [c for c in result[0]["content"] if c.get("type") == "text"]
    assert any("path" in part["text"] for part in text_parts)


def test_computer_image_adds_followup_text(interpreter):
    import base64

    png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    messages = [
        {
            "role": "computer",
            "type": "image",
            "format": "base64.png",
            "content": png,
        }
    ]
    result = convert_to_openai_messages(
        messages, vision=True, shrink_images=False, interpreter=interpreter
    )
    text_parts = [c for c in result[0]["content"] if c.get("type") == "text"]
    assert any("tool output" in part["text"] for part in text_parts)


def test_error_type_ignored(interpreter):
    messages = [
        {"role": "user", "type": "error", "content": "oops"},
        {"role": "user", "type": "message", "content": "hi"},
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "user", "content": "User: hi"}]


def test_always_apply_user_message_template(interpreter):
    interpreter.always_apply_user_message_template = True
    messages = [
        {"role": "user", "type": "message", "content": "first"},
        {"role": "user", "type": "message", "content": "second"},
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result[0]["content"] == "User: first"
    assert result[1]["content"] == "User: second"


def test_function_calling_false_merges_same_role(interpreter):
    messages = [
        {"role": "assistant", "type": "message", "content": "part one"},
        {"role": "assistant", "type": "message", "content": "part two"},
    ]
    result = convert_to_openai_messages(
        messages, function_calling=False, interpreter=interpreter
    )
    assert len(result) == 1
    assert "part one" in result[0]["content"]
    assert "part two" in result[0]["content"]


def test_image_missing_format_raises(interpreter):
    with pytest.raises(Exception, match="format"):
        convert_to_openai_messages(
            [{"role": "user", "type": "image", "content": "data"}],
            vision=True,
            interpreter=interpreter,
        )
