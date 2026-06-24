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
