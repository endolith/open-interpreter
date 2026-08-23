import json
from types import SimpleNamespace

import pytest

from interpreter.core.llm.utils.convert_to_openai_messages import (
    convert_to_openai_messages,
)


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
    """Plain assistant messages pass through as role/content pairs unchanged."""
    messages = [{"role": "assistant", "type": "message", "content": "Hello"}]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "assistant", "content": "Hello"}]


def test_last_user_message_gets_template(interpreter):
    """The last user message is wrapped with the interpreter's user_message_template."""
    messages = [{"role": "user", "type": "message", "content": "Hello"}]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "user", "content": "User: Hello"}]


def test_code_with_function_calling(interpreter):
    """Code messages become assistant function_call payloads when function_calling is enabled."""
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
    """Code messages become markdown fenced blocks when function_calling is disabled."""
    messages = [
        {"role": "assistant", "type": "code", "format": "python", "content": "1+1"}
    ]
    result = convert_to_openai_messages(
        messages, function_calling=False, interpreter=interpreter
    )
    assert result[0]["content"] == "```python\n1+1\n```"


def test_console_output_function_role(interpreter):
    """Console output is sent as a function-role execute result when function_calling is enabled."""
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
    """Whitespace-only console output is replaced with a 'No output' placeholder for the LLM."""
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
    """Messages addressed to a non-assistant recipient are omitted from the OpenAI payload."""
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
    """Image description text is forwarded as a user message when vision is disabled."""
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
    """File messages are converted to user messages containing the file path."""
    messages = [{"role": "user", "type": "file", "content": "/path/to/file.txt"}]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "user", "content": "/path/to/file.txt"}]


def test_unknown_type_raises(interpreter):
    """An unrecognized message type raises so conversion bugs surface instead of silently dropping data."""
    with pytest.raises(Exception, match="Unable to convert"):
        convert_to_openai_messages(
            [{"role": "user", "type": "unknown", "content": "x"}],
            interpreter=interpreter,
        )


def test_console_output_assistant_sender(interpreter):
    """When code_output_sender is assistant, console output is formatted as assistant text, not function role."""
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
    assert result[0]["content"] == "```output\nresult\n```"


def test_vision_false_skips_base64_image(interpreter):
    """Base64 images are omitted entirely from the payload when vision support is disabled."""
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
    """Image path messages are read from disk and encoded as image_url parts when vision is enabled."""
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
    """Computer-origin images include explanatory text so the model knows the image is tool output."""
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
    """Error-type messages are dropped so they do not pollute the LLM conversation history."""
    messages = [
        {"role": "user", "type": "error", "content": "oops"},
        {"role": "user", "type": "message", "content": "hi"},
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "user", "content": "User: hi"}]


def test_always_apply_user_message_template(interpreter):
    """When always_apply_user_message_template is set, every user message gets the template, not just the last."""
    interpreter.always_apply_user_message_template = True
    messages = [
        {"role": "user", "type": "message", "content": "first"},
        {"role": "user", "type": "message", "content": "second"},
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result[0]["content"] == "User: first"
    assert result[1]["content"] == "User: second"


def test_function_calling_false_merges_same_role(interpreter):
    """Without function calling, consecutive same-role messages are merged into one content string."""
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
    """Image messages without a format field raise because the encoder cannot choose an encoding."""
    with pytest.raises(Exception, match="Format of the image is not specified"):
        convert_to_openai_messages(
            [{"role": "user", "type": "image", "content": "data"}],
            vision=True,
            interpreter=interpreter,
        )


def test_console_output_missing_content_does_not_crash(interpreter):
    """Console output with no content key is treated as 'No output' instead of raising."""
    messages = [{"role": "computer", "type": "console", "format": "output"}]
    result = convert_to_openai_messages(
        messages, function_calling=True, interpreter=interpreter
    )
    assert result[0]["role"] == "function"
    assert result[0]["content"] == "No output"


def test_console_output_missing_content_no_function_calling(interpreter):
    """Console output with no content key does not raise when function_calling is off.

    The content default is applied before the function_calling branch, so the
    plain (user-sender) path also reaches message["content"].strip() safely.
    """
    messages = [{"role": "computer", "type": "console", "format": "output"}]
    result = convert_to_openai_messages(
        messages, function_calling=False, interpreter=interpreter
    )
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "(no output)"

def test_console_output_non_string_content_is_coerced(interpreter):
    """Non-string console output (e.g. an int) is coerced to a string before sending."""
    interpreter.debug = True
    messages = [
        {"role": "computer", "type": "console", "format": "output", "content": 42}
    ]
    result = convert_to_openai_messages(
        messages, function_calling=True, interpreter=interpreter
    )
    assert result[0]["content"] == "42"


def test_code_output_sender_user_applies_template(interpreter):
    """With code_output_sender='user', non-empty console output is wrapped in the code_output_template."""
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
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Output:\nresult"


def test_code_output_sender_user_uses_empty_template(interpreter):
    """With code_output_sender='user', empty console output uses the empty_code_output_template."""
    messages = [
        {"role": "computer", "type": "console", "format": "output", "content": "  "}
    ]
    result = convert_to_openai_messages(
        messages, function_calling=False, interpreter=interpreter
    )
    assert result[0]["content"] == "(no output)"


def test_base64_image_without_dot_defaults_to_png(interpreter):
    """A base64 image format without a file extension is encoded as PNG."""
    import base64

    png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    messages = [
        {"role": "user", "type": "image", "format": "base64", "content": png}
    ]
    result = convert_to_openai_messages(
        messages, vision=True, shrink_images=False, interpreter=interpreter
    )
    url = result[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_unrecognized_image_format_raises(interpreter):
    """An image with a non-base64, non-path format raises a descriptive error."""
    with pytest.raises(Exception, match="Unrecognized image format"):
        convert_to_openai_messages(
            [{"role": "user", "type": "image", "format": "weird", "content": "x"}],
            vision=True,
            interpreter=interpreter,
        )


def test_large_image_is_shrunk_below_5mb(interpreter):
    """Images larger than 5MB are resized down until the data URI fits the 5MB budget."""
    import base64
    import io
    import os

    from PIL import Image

    noise = os.urandom(2000 * 2000 * 3)
    img = Image.frombytes("RGB", (2000, 2000), noise)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    result = convert_to_openai_messages(
        [{"role": "user", "type": "image", "format": "base64.png", "content": b64}],
        vision=True,
        shrink_images=True,
        interpreter=interpreter,
    )
    url = result[0]["content"][0]["image_url"]["url"]
    assert len(url) < 5 * 1024 * 1024


def test_computer_path_image_appends_path_to_existing_text(tmp_path, interpreter):
    """A computer image loaded from a path appends the path note to the existing tool-output text."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01")
    messages = [
        {"role": "computer", "type": "image", "format": "path", "content": str(img)}
    ]
    result = convert_to_openai_messages(
        messages, vision=True, shrink_images=False, interpreter=interpreter
    )
    text_parts = [c["text"] for c in result[0]["content"] if c.get("type") == "text"]
    assert len(text_parts) == 1
    assert "last tool output" in text_parts[0]
    assert "at this path" in text_parts[0]
    assert str(img) in text_parts[0]


def test_merge_flushes_on_role_change(interpreter):
    """Without function calling, a role change flushes the accumulated same-role messages."""
    messages = [
        {"role": "assistant", "type": "message", "content": "part one"},
        {"role": "user", "type": "message", "content": "question"},
    ]
    result = convert_to_openai_messages(
        messages, function_calling=False, interpreter=interpreter
    )
    assert [m["role"] for m in result] == ["assistant", "user"]
    assert result[0]["content"] == "part one"
    assert result[1]["content"] == "User: question"


def test_merge_flushes_on_non_string_content(interpreter):
    """A non-string message (e.g. an image) interrupts and flushes pending text messages."""
    import base64

    png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    messages = [
        {"role": "assistant", "type": "message", "content": "text before"},
        {"role": "user", "type": "image", "format": "base64.png", "content": png},
        {"role": "assistant", "type": "message", "content": "text after"},
    ]
    result = convert_to_openai_messages(
        messages, vision=True, function_calling=False, interpreter=interpreter
    )
    assert [m["role"] for m in result] == ["assistant", "user", "assistant"]
    assert result[0]["content"] == "text before"
    assert result[1]["content"][0]["type"] == "image_url"


def test_empty_message_list_returns_empty(interpreter):
    """An empty input list yields an empty OpenAI payload rather than raising.

    Callers that build messages incrementally may hit this function before any
    message is appended; it must be a safe no-op."""
    result = convert_to_openai_messages([], interpreter=interpreter)
    assert result == []


def test_middle_user_message_is_not_templated(interpreter):
    """Only the last user message gets the template when always_apply is off.

    The template is meant to tag the latest user request, so earlier user
    messages must pass through verbatim."""
    messages = [
        {"role": "user", "type": "message", "content": "first"},
        {"role": "user", "type": "message", "content": "second"},
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "User: second"},
    ]


def test_recipient_assistant_explicitly_kept(interpreter):
    """A message explicitly addressed to the assistant is kept in the payload.

    The recipient filter only drops non-assistant recipients, so an explicit
    'assistant' recipient must survive conversion."""
    messages = [
        {
            "role": "user",
            "type": "message",
            "content": "for you",
            "recipient": "assistant",
        }
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "user", "content": "User: for you"}]


def test_whitespace_around_text_content_is_stripped(interpreter):
    """Leading and trailing whitespace is stripped from string content.

    Models are sensitive to stray whitespace in the payload; every string
    content is normalized before it is sent."""
    messages = [
        {"role": "assistant", "type": "message", "content": "  Hello world  "}
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "assistant", "content": "Hello world"}]


def test_merge_joins_same_role_messages_with_newline(interpreter):
    """Without function calling, a flushed same-role group joins with newlines.

    The role-change flush joins accumulated messages with a newline (the final
    flush uses a space), so a two-message assistant group becomes one block."""
    messages = [
        {"role": "assistant", "type": "message", "content": "part one"},
        {"role": "assistant", "type": "message", "content": "part two"},
        {"role": "user", "type": "message", "content": "question"},
    ]
    result = convert_to_openai_messages(
        messages, function_calling=False, interpreter=interpreter
    )
    assert len(result) == 2
    assert [m["role"] for m in result] == ["assistant", "user"]
    assert result[0]["content"] == "part one\npart two"
    assert result[1]["content"] == "User: question"


def test_console_message_with_non_output_format_raises(interpreter):
    """A console message whose format is not 'output' is un-convertible.

    The function only knows how to encode console 'output'; any other format
    falls through to the unknown-type branch and raises."""
    with pytest.raises(Exception, match="Unable to convert"):
        convert_to_openai_messages(
            [
                {
                    "role": "computer",
                    "type": "console",
                    "format": "error",
                    "content": "boom",
                }
            ],
            interpreter=interpreter,
        )


def test_code_function_call_content_is_empty_string(interpreter):
    """A code message in function-calling mode carries an empty content field.

    OpenAI/Azure reject assistant messages with a function_call but no content
    key, so the converter must set content to an empty string."""
    messages = [
        {"role": "assistant", "type": "code", "format": "python", "content": "1+1"}
    ]
    result = convert_to_openai_messages(
        messages, function_calling=True, interpreter=interpreter
    )
    assert result[0]["content"] == ""
    assert result[0]["function_call"]["name"] == "execute"


def test_base64_image_with_dot_extension_uses_that_extension(interpreter):
    """A base64 image format with a dotted extension is encoded with it.

    The MIME type is derived from the extension in the format string (e.g.
    'base64.jpeg' -> image/jpeg), not hardcoded to png."""
    import base64

    png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    messages = [
        {"role": "user", "type": "image", "format": "base64.jpeg", "content": png}
    ]
    result = convert_to_openai_messages(
        messages, vision=True, shrink_images=False, interpreter=interpreter
    )
    url = result[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_image_description_passes_through_with_vision_enabled(interpreter):
    """Description images pass through unchanged even when vision is on.

    A description is already text, so enabling vision must not route it through
    the base64/path image encoding."""
    messages = [
        {
            "role": "user",
            "type": "image",
            "format": "description",
            "content": "A gradient image",
        }
    ]
    result = convert_to_openai_messages(messages, vision=True, interpreter=interpreter)
    assert result == [{"role": "user", "content": "A gradient image"}]


def test_image_description_normalizes_computer_role(interpreter):
    """A description image is emitted as a user message, not with a 'computer' role.

    The source message role can be 'computer' (screenshot tool output), but
    'computer' is not a valid OpenAI Chat Completions role, so the description
    branch normalizes it to 'user'."""
    messages = [
        {
            "role": "computer",
            "type": "image",
            "format": "description",
            "content": "tool output described",
        }
    ]
    result = convert_to_openai_messages(messages, vision=True, interpreter=interpreter)
    assert result == [{"role": "user", "content": "tool output described"}]
