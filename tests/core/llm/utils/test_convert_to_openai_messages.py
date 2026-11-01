import base64
import io
import json
import os
import sys
from types import SimpleNamespace

import pytest

from interpreter.core.llm.utils.convert_to_openai_messages import (
    convert_to_openai_messages as _convert_to_openai_messages,
)

import re

_TS_PREFIX = re.compile(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] ")


def convert_to_openai_messages(*args, **kwargs):
    """Converter wrapper that drops user-message timestamp prefixes.

    Timestamps are an orthogonal feature (`sent_at`); these tests pin
    conversion semantics, so the prefix is removed before comparison.
    """
    result = _convert_to_openai_messages(*args, **kwargs)
    for message in result:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _TS_PREFIX.sub("", content)
        elif isinstance(content, list):
            message["content"] = [
                part
                for part in content
                if not (
                    part.get("type") == "text"
                    and _TS_PREFIX.fullmatch(part.get("text", ""))
                )
            ]
    return result


def _image_url_part(message):
    """The image_url dict from a converted message, ignoring any timestamp text part."""
    for part in message["content"]:
        if part.get("type") == "image_url":
            return part["image_url"]
    raise AssertionError(f"no image_url part in {message['content']!r}")

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
    assert any(p.get("type") == "image_url" for p in result[0]["content"])
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
    url = _image_url_part(result[0])["url"]
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
    url = _image_url_part(result[0])["url"]
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
    assert any(p.get("type") == "image_url" for p in result[1]["content"])


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
    url = _image_url_part(result[0])["url"]
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


def _png_bytes():
    """A small valid 2x2 PNG, for tests that need a decodable image."""
    from PIL import Image

    buffered = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffered, format="png")
    return buffered.getvalue()


def _oversized_png_path(tmp_path):
    """A PNG whose base64 data URL just exceeds the 5 MB shrink threshold (and is still decodable)."""
    import os

    from PIL import Image

    width = height = 1118
    img = Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    buffered = io.BytesIO()
    img.save(buffered, format="png", compress_level=0)
    png = buffered.getvalue()
    # Pad after IEND so the base64 length lands in (5*2^20, 5*(1025/1024)^2*2^20)
    # bytes: large enough that the original code shrinks it, small enough that
    # a mutant computing MB with 1025*1024 does not.
    png = png + os.urandom(3932115 - len(png))
    path = tmp_path / "big.png"
    path.write_bytes(png)
    return path, width


def test_recipient_skip_continues_later_messages(interpreter):
    """Messages addressed to the user are skipped, but later messages are still processed."""
    messages = [
        {
            "role": "assistant",
            "type": "message",
            "content": "for the user",
            "recipient": "user",
        },
        {"role": "assistant", "type": "message", "content": "Visible"},
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "assistant", "content": "Visible"}]


def test_error_message_skipped_but_later_processed(interpreter):
    """Error messages are skipped, but later messages are still processed."""
    messages = [
        {"role": "system", "type": "error", "content": "boom"},
        {"role": "assistant", "type": "message", "content": "Visible"},
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "assistant", "content": "Visible"}]


def test_function_calling_false_merges_consecutive_messages(interpreter):
    """With function calling off, consecutive same-role messages merge: earlier runs join with newlines, the final run with spaces."""
    messages = [
        {"role": "user", "type": "message", "content": "a"},
        {"role": "user", "type": "message", "content": "b"},
        {"role": "assistant", "type": "message", "content": "x"},
        {"role": "user", "type": "message", "content": "c"},
        {"role": "user", "type": "message", "content": "d"},
    ]
    result = convert_to_openai_messages(
        messages, interpreter=interpreter, function_calling=False
    )
    assert result == [
        {"role": "user", "content": "a\nb"},
        {"role": "assistant", "content": "x"},
        # The last user message gets the interpreter's user_message_template.
        {"role": "user", "content": "c User: d"},
    ]


def test_vision_defaults_to_false_skips_images(interpreter, tmp_path):
    """Without vision enabled, image messages are dropped and later messages still appear."""
    path = tmp_path / "pic.png"
    path.write_bytes(_png_bytes())
    messages = [
        {"role": "computer", "type": "image", "format": "path", "content": str(path)},
        {"role": "assistant", "type": "message", "content": "Visible"},
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "assistant", "content": "Visible"}]


def test_base64_image_extension_from_format(interpreter):
    """For base64 images, the data URL's extension comes from the last dot-separated segment of the format."""
    fake_b64 = base64.b64encode(_png_bytes()).decode()
    messages = [
        {
            "role": "computer",
            "type": "image",
            "format": "base64.image.png",
            "content": fake_b64,
        }
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter, vision=True)
    url = _image_url_part(result[0])["url"]
    assert url.startswith("data:image/png;base64,")


def test_path_image_extension_from_filename(interpreter, tmp_path):
    """For image paths, the data URL's extension comes from the file name and the base64 payload is the file's bytes."""
    path = tmp_path / "pic.v2.png"
    path.write_bytes(_png_bytes())
    messages = [
        {"role": "user", "type": "image", "format": "path", "content": str(path)}
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter, vision=True)
    url = _image_url_part(result[0])["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == _png_bytes()


def test_computer_image_gets_detail_low_and_prompt_text(interpreter):
    """A computer-sourced base64 image becomes an image_url part with detail=low plus the tool-output prompt text."""
    fake_b64 = base64.b64encode(_png_bytes()).decode()
    messages = [
        {
            "role": "computer",
            "type": "image",
            "format": "base64.png",
            "content": fake_b64,
        }
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter, vision=True)
    assert result == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + fake_b64,
                        "detail": "low",
                    },
                },
                {
                    "type": "text",
                    "text": "This image is the result of the last tool output. What does it mean / are we done?",
                },
            ],
        }
    ]


def test_path_image_includes_path_text(interpreter, tmp_path):
    """A user-sourced image path message includes the path in a text part."""
    path = tmp_path / "shot.png"
    path.write_bytes(_png_bytes())
    messages = [
        {"role": "user", "type": "image", "format": "path", "content": str(path)}
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter, vision=True)
    assert result[0]["content"][1] == {
        "type": "text",
        "text": "This image is at this path: " + str(path),
    }


def test_computer_path_image_appends_path_text(interpreter, tmp_path):
    """A computer-sourced image path message appends the path to the existing prompt text."""
    path = tmp_path / "shot.png"
    path.write_bytes(_png_bytes())
    messages = [
        {"role": "computer", "type": "image", "format": "path", "content": str(path)}
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter, vision=True)
    assert result[0]["content"][1]["text"] == (
        "This image is the result of the last tool output. What does it mean / are we done?\nThis image is at this path: "
        + str(path)
    )


def test_oversized_image_is_shrunk_below_limit(interpreter, tmp_path):
    """An image whose data URL exceeds 5 MB is resized (shrink_images defaults to True) by the expected scale factor."""
    from PIL import Image

    path, width = _oversized_png_path(tmp_path)
    original_b64 = base64.b64encode(path.read_bytes()).decode()
    original_content = "data:image/png;base64," + original_b64

    messages = [
        {"role": "computer", "type": "image", "format": "path", "content": str(path)}
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter, vision=True)

    url = _image_url_part(result[0])["url"]
    # Shrinking happened at all.
    assert len(url) < len(original_content)
    # The scale factor matches (4.9 / size_mb) ** 0.5 applied once.
    size_mb = sys.getsizeof(original_content) / (1024 * 1024)
    expected_width = int(width * (4.9 / size_mb) ** 0.5)
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert img.width == expected_width
def test_merge_flushes_accumulated_text_on_image(interpreter, tmp_path):
    """When an image message (list content) interrupts accumulated text messages, the buffered text flushes with newline joins."""
    path = tmp_path / "pic.png"
    path.write_bytes(_png_bytes())
    messages = [
        {"role": "user", "type": "message", "content": "a"},
        {"role": "user", "type": "message", "content": "b"},
        {"role": "computer", "type": "image", "format": "path", "content": str(path)},
    ]
    result = convert_to_openai_messages(
        messages, interpreter=interpreter, vision=True, function_calling=False
    )
    # "b" is the last user text message, so it gets the template before joining.
    assert result[0] == {"role": "user", "content": "a\nUser: b"}
    assert any(p.get("type") == "image_url" for p in result[1]["content"])


def test_non_string_console_content_is_coerced_to_str(interpreter):
    """A function-calling console chunk with non-str content is str()-ed before use.

    A list payload would otherwise reach .strip() and raise, so it is str()-ed
    first. (Changing the checked expression to type(None) is an equivalent
    mutant here: non-str content coerces either way, and str content is
    unchanged.)
    """
    interpreter.debug = True
    message = {
        "role": "assistant",
        "type": "console",
        "format": "output",
        "content": ["line one", "line two"],
    }

    result = convert_to_openai_messages(
        [message], function_calling=True, interpreter=interpreter
    )

    assert result[0]["content"] == str(["line one", "line two"])


def test_image_message_without_format_key_raises_descriptive_error(interpreter):
    """An image chunk with no "format" key fails with a descriptive error.

    The guard is fixed: message.get("format", "") no longer raises KeyError
    before the "format not in message" check, so the intended "Format of the
    image is not specified." message surfaces instead of a bare KeyError.
    """
    message = {"role": "user", "type": "image", "content": "/tmp/pic.png"}

    with pytest.raises(Exception, match="Format of the image is not specified"):
        convert_to_openai_messages([message], vision=True, interpreter=interpreter)


def test_reasoning_content_passed_back_verbatim(interpreter):
    """Stored reasoning_content is echoed on the assistant message so thinking-model follow-ups are accepted (issue #215)."""
    messages = [
        {
            "role": "assistant",
            "type": "message",
            "content": "hi",
            "reasoning_content": "thought one",
        }
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [
        {"role": "assistant", "content": "hi", "reasoning_content": "thought one"}
    ]


def test_message_without_reasoning_omits_the_key(interpreter):
    """Messages without reasoning stay exactly role/content (issue #215)."""
    messages = [{"role": "assistant", "type": "message", "content": "Hello"}]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result == [{"role": "assistant", "content": "Hello"}]


def test_function_calling_false_preserves_reasoning(interpreter):
    """The combine path must keep reasoning_content, not just content."""
    messages = [
        {
            "role": "assistant",
            "type": "message",
            "content": "a",
            "reasoning_content": "r1",
        },
        {
            "role": "assistant",
            "type": "message",
            "content": "b",
            "reasoning_content": "r2",
        },
    ]

    result = convert_to_openai_messages(
        messages, function_calling=False, interpreter=interpreter
    )

    assert result == [
        {
            "role": "assistant",
            "content": "a b",
            "reasoning_content": "r1\nr2",
        }
    ]


def test_code_message_carries_reasoning(interpreter):
    """Reasoning attached to a code message survives conversion."""
    messages = [
        {
            "role": "assistant",
            "type": "code",
            "format": "python",
            "content": "1+1",
            "reasoning_content": "thinking",
        }
    ]

    result = convert_to_openai_messages(messages, interpreter=interpreter)

    assert result[0]["reasoning_content"] == "thinking"


class _FakeInterpreter:
    """Minimal stand-in for the interpreter attributes convert_to_openai_messages touches."""

    always_apply_user_message_template = False
    user_message_template = "{content}"
    code_output_sender = "function"
    empty_code_output_template = ""
    debug = False


def test_computer_role_description_image_becomes_user():
    messages = [
        {
            "role": "computer",
            "type": "image",
            "format": "description",
            "content": "A screenshot of bold text.",
            "recipient": "assistant",
        }
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=None)
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert "screenshot" in out[0]["content"]


def test_html_skips_png_when_not_vision():
    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    interpreter.llm.supports_vision = False
    html = interpreter.terminal.get_language("html")
    html_lang = html(interpreter)
    chunks = list(html_lang.run("<b>edited</b>"))
    image_chunks = [c for c in chunks if c.get("type") == "image"]
    assert image_chunks == []
    assistant_chunks = [c for c in chunks if c.get("recipient") == "assistant" and c.get("type") == "console"]
    assert any("```html" in c.get("content", "") for c in assistant_chunks)


def test_react_incompatible_code_reports_error_without_crashing():
    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    react = interpreter.terminal.get_language("react")
    react_lang = react(interpreter)
    chunks = list(react_lang.run("import React from 'react'"))
    assert chunks
    assert chunks[0]["type"] == "console"
    assert "React format not supported" in chunks[0]["content"]
    assert "require" in chunks[0]["content"] or "import" in chunks[0]["content"]


def test_react_compatible_code_yields_html_for_user():
    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    interpreter.llm.supports_vision = False
    react = interpreter.terminal.get_language("react")
    react_lang = react(interpreter)
    code = 'ReactDOM.render(<h1>Hello, edit!</h1>, document.getElementById("root"));'
    chunks = list(react_lang.run(code))
    user_html = [c for c in chunks if c.get("recipient") == "user" and c.get("format") == "html"]
    assert len(user_html) == 1
    assert "Hello, edit!" in user_html[0]["content"]
    assert 'type="text/babel"' in user_html[0]["content"]


def test_reasoning_content_propagates_to_all_tool_calls_in_multi_code_turn():
    """Every tool-call message in a single turn must carry the turn's reasoning_content.

    DeepSeek's thinking mode returns a 400 if any assistant tool-call message in the
    history lacks reasoning_content. When one response produces several code blocks
    separated by tool output, the old code dropped the pending reasoning at the first
    tool/function message, leaving later tool calls unadorned. This verifies the
    reasoning survives intervening tool responses so no tool-call message is bare.
    """
    messages = [
        {"role": "user", "type": "message", "content": "do A and B"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "Plan. \n\n"},
        {"role": "assistant", "type": "message", "content": "Doing A."},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('B')"},
        {"role": "computer", "type": "console", "format": "output", "content": "B"},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_FakeInterpreter())
    function_calls = [m for m in out if "function_call" in m]
    assert len(function_calls) == 2
    assert all(m["reasoning_content"] == "Plan. \n\n" for m in function_calls)


def test_tool_loop_reasoning_replaced_per_llm_call():
    """A new reasoning block after tool output replaces, not appends to, the old one.

    The tool loop makes a fresh LLM call per tool, so each call's reasoning is stored as
    its own reasoning message separated by tool output. The pending reasoning must reset
    for the new call (otherwise the second tool-call message would receive the first
    call's reasoning plus the second's, polluting history with stale thoughts).
    """
    messages = [
        {"role": "user", "type": "message", "content": "do A then B"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "R1. \n\n"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "R2. \n\n"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('B')"},
        {"role": "computer", "type": "console", "format": "output", "content": "B"},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_FakeInterpreter())
    function_calls = [m for m in out if "function_call" in m]
    assert len(function_calls) == 2
    assert function_calls[0]["reasoning_content"] == "R1. \n\n"
    assert function_calls[1]["reasoning_content"] == "R2. \n\n"


def test_post_stream_reasoning_backfills_earlier_assistant_messages():
    """Reasoning stored after the code it belongs to must still reach that tool-call message.

    Some providers deliver reasoning_content only once the stream completes, and the
    reasoning message can end up stored after the content/code messages in history.
    Without backfilling, the tool-call message would be sent without reasoning_content
    and DeepSeek would reject the request with a 400.
    """
    messages = [
        {"role": "user", "type": "message", "content": "compute"},
        {"role": "assistant", "type": "message", "content": "Running now."},
        {"role": "assistant", "type": "code", "format": "python", "content": "print(2+2)"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "Computed 2+2. \n\n"},
        {"role": "computer", "type": "console", "format": "output", "content": "4"},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_FakeInterpreter())
    function_calls = [m for m in out if "function_call" in m]
    assert len(function_calls) == 1
    assert function_calls[0]["reasoning_content"] == "Computed 2+2. \n\n"


def test_whitespace_only_assistant_separator_is_dropped():
    """A stored loop-mode separator must never reach the API as an empty assistant message.

    respond() stores an assistant message containing only "\\n\\n" as a visual separator
    between turns when loop mode auto-continues. convert_to_openai_messages strips it to
    an empty string, yielding an assistant message with neither content nor tool_calls.
    DeepSeek (and OpenRouter's BYOK relay for it) rejects such a message with a 400
    ("Invalid assistant message: content or tool_calls must be set"). The separator
    carries no information to the model, so it must be dropped from the request.
    """
    messages = [
        {"role": "user", "type": "message", "content": "hi"},
        {"role": "assistant", "type": "message", "content": "Let me do that."},
        {"role": "assistant", "type": "message", "content": "\n\n"},
        {"role": "user", "type": "message", "content": "Proceed."},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_FakeInterpreter())
    for m in out:
        if m["role"] == "assistant":
            assert m.get("tool_calls") or str(m.get("content", "")).strip(), (
                f"assistant message with neither content nor tool_calls leaked: {m!r}"
            )
    assert len(out) == 3


def test_whitespace_separator_does_not_break_reasoning_propagation():
    """Dropping a whitespace separator must not sever the turn's reasoning_content chain.

    The loop separator can be stored between a reasoning block and the assistant message
    it belongs to. Even though the separator is dropped, the following tool-call message
    must still receive the turn's reasoning_content or DeepSeek returns a 400.
    """
    messages = [
        {"role": "user", "type": "message", "content": "do A"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "Plan. \n\n"},
        {"role": "assistant", "type": "message", "content": "\n\n"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_FakeInterpreter())
    function_calls = [m for m in out if "function_call" in m]
    assert len(function_calls) == 1
    assert function_calls[0]["reasoning_content"] == "Plan. \n\n"


def test_user_message_boundary_resets_pending_reasoning():
    """A new user turn must not inherit the previous turn's reasoning_content.

    Reasoning belongs to the turn that produced it. After a user message, the next
    tool-call message carries no reasoning_content (the API-layer fallback pads it with
    "" for turns where the model did not think), so a stale reasoning value must not
    leak across the turn boundary.
    """
    messages = [
        {"role": "user", "type": "message", "content": "hi"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "Greet. \n\n"},
        {"role": "assistant", "type": "message", "content": "Hello!"},
        {"role": "user", "type": "message", "content": "compute"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print(1)"},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_FakeInterpreter())
    function_calls = [m for m in out if "function_call" in m]
    assert len(function_calls) == 1
    assert "reasoning_content" not in function_calls[0]


def _make_image_bytes(image_format="PNG", color=(10, 20, 30)):
    """Return the raw bytes of a tiny image saved in the given PIL format."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format=image_format)
    return buffer.getvalue()


def _path_image_message(raw_bytes, extension):
    """Write raw_bytes to a temp file with the given extension and return its path."""
    import tempfile

    fd, path = tempfile.mkstemp(suffix=f".{extension}")
    with os.fdopen(fd, "wb") as file:
        file.write(raw_bytes)
    return path


def _image_url(out):
    """Extract the data URL from the first image_url content part in converted messages."""
    for message in out:
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image_url":
                return part["image_url"]["url"]
    raise AssertionError("no image_url part found in converted messages")


def _convert_image_path(path, **kwargs):
    """Convert a single path-based image message, removing the temp file afterwards."""
    try:
        return convert_to_openai_messages(
            [{"role": "user", "type": "image", "format": "path", "content": path}],
            function_calling=True,
            vision=True,
            shrink_images=False,
            interpreter=_FakeInterpreter(),
            **kwargs,
        )
    finally:
        os.remove(path)


def test_bmp_path_is_retranscoded_to_png():
    """A correctly-labeled .bmp must be sent as a decodable PNG data URL.

    No major vision API (OpenAI, Claude, DeepSeek, Gemini) accepts BMP bytes;
    providers return a 400 "unsupported image" error. The converter sniffs the
    real format and re-encodes BMP losslessly to PNG so the request reaches the
    model instead of being rejected.
    """
    out = _convert_image_path(_path_image_message(_make_image_bytes("BMP"), "bmp"))
    url = _image_url(out)
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split("base64,")[1]).startswith(b"\x89PNG")


def test_bmp_misnamed_png_is_sent_as_real_png():
    """A BMP carrying a .png extension must not be sent as image/png holding BMP bytes.

    Regression case behind a real provider 400: the file was a BMP renamed to
    .png, so the old code built ``data:image/png;base64,<BMP bytes>`` which the
    API rejected. The bytes are sniffed, the BMP re-encoded to PNG, and the data
    URL actually decodes as a PNG.
    """
    raw = _make_image_bytes("BMP")
    out = _convert_image_path(_path_image_message(raw, "png"))
    url = _image_url(out)
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split("base64,")[1]).startswith(b"\x89PNG")
    assert base64.b64decode(url.split("base64,")[1]) != raw


def test_base64_bmp_message_is_retranscoded_to_png():
    """Inline base64.bmp images get the same normalization as path-based ones.

    base64 image messages (e.g. from the computer/display toolboxes) share the
    same converter branch, so a BMP payload must also be re-encoded to PNG rather
    than sent as an unsupported data:image/bmp URL.
    """
    out = convert_to_openai_messages(
        [
            {
                "role": "computer",
                "type": "image",
                "format": "base64.bmp",
                "content": base64.b64encode(_make_image_bytes("BMP")).decode(),
            }
        ],
        function_calling=True,
        vision=True,
        shrink_images=False,
        interpreter=_FakeInterpreter(),
    )
    url = _image_url(out)
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split("base64,")[1]).startswith(b"\x89PNG")


def test_supported_format_paths_pass_through_unchanged():
    """PNG/JPEG/WebP/GIF bytes must reach the API byte-for-byte unchanged.

    These are the formats every major vision API accepts; re-encoding them would
    waste tokens and could alter fidelity. The data URL must carry the *real*
    format in its MIME type, correcting a misleading extension (e.g. a JPEG saved
    as .png is labeled image/jpeg) while keeping the original bytes.
    """
    from PIL import features

    webp_supported = features.check("webp")
    cases = [
        ("PNG", "png", "image/png"),
        ("JPEG", "jpg", "image/jpeg"),
        ("JPEG", "png", "image/jpeg"),
        ("GIF", "gif", "image/gif"),
    ]
    if webp_supported:
        cases.append(("WEBP", "webp", "image/webp"))
    for image_format, extension, mime in cases:
        raw = _make_image_bytes(image_format)
        out = _convert_image_path(_path_image_message(raw, extension))
        url = _image_url(out)
        assert url.startswith(f"data:{mime};base64,"), (image_format, url[:40])
        assert url.split("base64,")[1] == base64.b64encode(raw).decode()


def test_undecodable_bytes_fall_back_to_declared_format():
    """Corrupt/non-image data is passed through with its declared format, not crashed.

    Pillow cannot inspect garbage bytes; the converter must not raise. The
    request then fails at the provider with a clear 400 instead of an opaque
    local error, and the surrounding path handling can surface the filename for
    diagnosis.
    """
    out = _convert_image_path(_path_image_message(b"this is not an image at all", "png"))
    url = _image_url(out)
    assert url.startswith("data:image/png;base64,")
    assert url.split("base64,")[1] == base64.b64encode(b"this is not an image at all").decode()


class _FakeStrictDeepSeekInterpreter(_FakeInterpreter):
    """Same stand-in, but routed through a strict DeepSeek-via-OpenRouter model."""

    class llm:
        model = "openrouter/~deepseek/deepseek-v4-flash-latest"


class _FakeGoDeepSeekInterpreter(_FakeInterpreter):
    """Stand-in for a DeepSeek model reached via the opencode_go/ prefix.

    Llm.load() used to rewrite that prefix to openai/<model> because the gateway
    speaks the OpenAI wire format, which hid the route from every gate that
    branches on the model name. The prefix is now kept, so the route is visible
    again and this is simply a DeepSeek model on it.
    """

    class llm:
        model = "opencode_go/deepseek-v4.1-flash"
        _is_opencode_go = True


DEEPSEEK_ROUTE_INTERPRETERS = [
    _FakeStrictDeepSeekInterpreter,
    _FakeGoDeepSeekInterpreter,
]


def test_deepseek_on_the_go_route_uses_modern_tool_calls():
    """A DeepSeek model on the Go route must get the modern shape.

    When the prefix was rewritten away, this route was never recognized as
    DeepSeek and kept emitting legacy function_call/function messages to a
    provider that rejects them. The gateway answered with a bodiless 400 on
    every tool-calling turn, and the legacy shape then had to be translated
    downstream, which is where the malformed tool_call/tool-response histories
    came from.
    """
    messages = [
        {"role": "user", "type": "message", "content": "run it"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
    ]
    out = convert_to_openai_messages(
        messages,
        function_calling=True,
        vision=False,
        interpreter=_FakeGoDeepSeekInterpreter(),
    )

    assert not any("function_call" in m for m in out), "the deprecated field is rejected by strict DeepSeek validators"
    assert not any(m.get("role") == "function" for m in out), (
        "a legacy function result leaves the tool message unpaired"
    )
    calls = [m for m in out if m.get("tool_calls")]
    assert len(calls) == 1
    outputs = [m for m in out if m.get("role") == "tool"]
    assert len(outputs) == 1
    assert outputs[0]["tool_call_id"] == calls[0]["tool_calls"][0]["id"]


def test_openai_routes_without_deepseek_keep_legacy_function_call():
    """Genuine OpenAI models on the same openai/ spelling are unaffected.

    The fix matches on the model name rather than the prefix, so an actual
    OpenAI model routed the same way must keep the legacy output it always had;
    some providers reject the modern shape or expect the old one.
    """

    class _OpenAIGpt(_FakeInterpreter):
        class llm:
            model = "openai/gpt-4o"

    messages = [
        {"role": "user", "type": "message", "content": "run it"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_OpenAIGpt())

    assert not any(m.get("tool_calls") for m in out)
    assert any(m.get("role") == "function" for m in out)


def test_strict_deepseek_route_uses_modern_tool_calls():
    """Strict DeepSeek routes must never see legacy function_call or function roles.

    With supports_functions enabled, code/edit turns convert to assistant
    messages with empty content plus a call payload, and console outputs to
    result messages. The legacy function_call/function shape is rejected by
    DeepSeek's strict validators (OpenRouter BYOK relay) with a 400
    ("Invalid assistant message: content or tool_calls must be set"), so for
    DeepSeek routes the converter must emit modern tool_calls and pair every
    tool output to its call via tool_call_id. Other models keep the legacy
    shape untouched.
    """
    messages = [
        {"role": "user", "type": "message", "content": "run it"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('B')"},
        {"role": "computer", "type": "console", "format": "output", "content": ""},
    ]
    out = convert_to_openai_messages(
        messages,
        function_calling=True,
        vision=False,
        interpreter=_FakeStrictDeepSeekInterpreter(),
    )
    assert not any("function_call" in m for m in out)
    assert not any(m.get("role") == "function" for m in out)
    calls = [m for m in out if m.get("tool_calls")]
    assert len(calls) == 2
    assert all(m["role"] == "assistant" for m in calls)
    call_ids = [tc["id"] for m in calls for tc in m["tool_calls"]]
    assert all(tc["type"] == "function" for m in calls for tc in m["tool_calls"])
    outputs = [m for m in out if m.get("role") == "tool"]
    assert len(outputs) == 2
    assert all(m.get("tool_call_id") in call_ids for m in outputs)
    # Empty code output still gets the explicit placeholder, now on a tool message.
    assert outputs[1]["content"] == "No output"
    # No assistant message goes out with neither content nor tool calls.
    for m in out:
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            assert str(m.get("content", "") or "").strip()


def test_strict_route_unpaired_code_falls_back_to_text():
    """A strict-route code turn with no following output must not become a bare call.

    History routinely holds code that never executed (interrupted loops,
    invalidated calls, code superseded by a user edit before running). Emitting
    it as tool_calls would leave a call id with no later matching tool output,
    which strict validators reject ("must be followed by tool messages ...").
    Such code falls back to a plain text block, exactly like text mode, so the
    request stays valid while keeping the code visible to the model.
    """
    messages = [
        {"role": "user", "type": "message", "content": "run it"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('B')"},
        {"role": "user", "type": "message", "content": "actually never mind that"},
    ]
    out = convert_to_openai_messages(
        messages,
        function_calling=True,
        vision=False,
        interpreter=_FakeStrictDeepSeekInterpreter(),
    )
    calls = [m for m in out if m.get("tool_calls")]
    assert len(calls) == 1
    call_ids = [tc["id"] for m in calls for tc in m["tool_calls"]]
    outputs = [m for m in out if m.get("role") == "tool"]
    assert len(outputs) == 1
    assert outputs[0]["tool_call_id"] in call_ids
    texts = [
        m
        for m in out
        if m.get("role") == "assistant" and not m.get("tool_calls") and "print('B')" in str(m.get("content", ""))
    ]
    assert len(texts) == 1
    assert "```python" in texts[0]["content"]


def test_non_deepseek_route_keeps_legacy_function_call():
    """The modern tool_calls emission must not change other models' requests.

    Only strict DeepSeek routes get the new shape; every other model keeps
    the legacy function_call/function output so OpenAI, Azure, and local
    providers see byte-identical requests to before.
    """
    messages = [
        {"role": "user", "type": "message", "content": "run it"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_FakeInterpreter())
    assert sum("function_call" in m for m in out) == 1
    assert not any(m.get("tool_calls") for m in out)
    assert any(m.get("role") == "function" and m.get("name") == "execute" for m in out)


class _FakeGoInterpreter(_FakeInterpreter):
    """Stand-in for any model reached through the opencode_go/ prefix.

    The prefix survives load, so the route is recognisable from the model name.
    These models deliberately do not contain "deepseek": an earlier fix keyed on
    that and so left every other Go model emitting the legacy shape the gateway
    rejects.
    """

    class llm:
        model = "opencode_go/space-bunny-free"
        _is_opencode_go = True


def test_opencode_go_route_uses_modern_tool_calls_for_any_model():
    """Every model on the Go route needs modern tool_calls, not just DeepSeek ones.

    The gateway rejects an assistant message carrying the deprecated
    `function_call` field with empty content and no `tool_calls` key, and it
    does so for all of the models it serves. The fix that added "openai/" to the
    DeepSeek name match was not enough: a model like space-bunny-free reaches
    this code as plain "openai/space-bunny-free" and got the legacy shape, so
    every tool-calling turn failed with a 400 from the gateway.
    """
    messages = [
        {"role": "user", "type": "message", "content": "run it"},
        {"role": "assistant", "type": "code", "format": "powershell", "content": "Get-Date"},
        {"role": "computer", "type": "console", "format": "output", "content": "ok"},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_FakeGoInterpreter())

    assert not any("function_call" in m for m in out)
    assert not any(m.get("role") == "function" for m in out)
    calls = [m for m in out if m.get("tool_calls")]
    outputs = [m for m in out if m.get("role") == "tool"]
    assert len(calls) == 1 and len(outputs) == 1
    assert outputs[0]["tool_call_id"] == calls[0]["tool_calls"][0]["id"]


def test_plain_openai_models_are_unaffected_by_the_go_route():
    """A real OpenAI route must not inherit the Go rule from the shared prefix.

    Both arrive as "openai/...", so the decision has to come from the route flag
    rather than the name -- otherwise every OpenAI request changes shape.
    """

    class _OpenAIGpt(_FakeInterpreter):
        class llm:
            model = "openai/gpt-4o"
            _is_opencode_go = False

    messages = [
        {"role": "user", "type": "message", "content": "run it"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_OpenAIGpt())

    assert not any(m.get("tool_calls") for m in out)
    assert any(m.get("role") == "function" for m in out)


def test_opencode_go_routes_use_modern_tool_calls_whatever_the_model():
    """The Go gateway rejects the legacy shape for every model it serves.

    _use_modern_tool_calls()'s own docstring already said so, but the condition
    only matched opencode_go/ routes with "deepseek" in the name -- any other
    model on the gateway kept emitting role: function, which the gateway rejects
    ("expected one of system|user|assistant|tool|developer", 400). Seen live on
    an opencode_go non-DeepSeek route: code in history, then every request 400s,
    retries included.
    """

    class _GoGpt(_FakeInterpreter):
        class llm:
            model = "opencode_go/openai/gpt-4o-mini"

    messages = [
        {"role": "user", "type": "message", "content": "run it"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_GoGpt())

    assert not any("function_call" in m for m in out)
    roles = [m.get("role") for m in out]
    assert "function" not in roles, f"legacy role rejected by the gateway: {roles}"
    calls = [m for m in out if m.get("tool_calls")]
    outputs = [m for m in out if m.get("role") == "tool"]
    assert len(calls) == 1 and len(outputs) == 1
    assert outputs[0]["tool_call_id"] == calls[0]["tool_calls"][0]["id"]


def test_openrouter_routes_without_deepseek_use_modern_tool_calls():
    """Every openrouter/ route gets the modern shape, DeepSeek or not.

    OpenRouter's request validator rejects the legacy role outright
    ("expected one of system|user|assistant|tool|developer", 400), so a single
    executed code block in history -- converted to a function message -- poisoned
    every later request on the route, retries included, with no recovery except
    dropping the history. Seen after %undo + retry on an openrouter non-DeepSeek
    route: the retry prompt appeared, but every retry 400d the same way.
    OpenRouter accepts tool_calls/tool universally and translates per upstream
    provider, so the legacy shape buys nothing there.
    """

    class _OpenRouterGpt(_FakeInterpreter):
        class llm:
            model = "openrouter/openai/gpt-4o-mini"

    messages = [
        {"role": "user", "type": "message", "content": "run it"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
        {"role": "user", "type": "message", "content": "ok another revision, try them again"},
    ]
    out = convert_to_openai_messages(messages, function_calling=True, vision=False, interpreter=_OpenRouterGpt())

    assert not any("function_call" in m for m in out)
    roles = [m.get("role") for m in out]
    assert "function" not in roles, f"legacy role rejected by OpenRouter: {roles}"
    assert all(r in ("system", "user", "assistant", "tool", "developer") for r in roles), roles
    calls = [m for m in out if m.get("tool_calls")]
    outputs = [m for m in out if m.get("role") == "tool"]
    assert len(calls) == 1 and len(outputs) == 1
    assert outputs[0]["tool_call_id"] == calls[0]["tool_calls"][0]["id"]



def test_repeated_outputs_become_one_tool_result():
    """Three console results for one html call collapse into one tool message.

    HTML/React yield "displayed", a failure note, and the source; the model saw
    three results for one call, which shifted the next call's result and made a
    stray python `pass` look like it produced the HTML source. Providers also
    reject duplicate tool_call_ids.
    """
    messages = [
        {"role": "user", "type": "message", "content": "do html"},
        {"role": "assistant", "type": "code", "format": "html", "content": "<h1>hi</h1>"},
        {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": "HTML is being displayed on the user's machine...",
            "recipient": "assistant",
        },
        {
            "role": "computer",
            "type": "code",
            "format": "html",
            "content": "<h1>hi</h1>",
            "recipient": "user",
        },
        {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": "Could not render a preview (the headless browser failed).",
            "recipient": "assistant",
        },
        {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": "HTML source shown to the user: ```html <h1>hi</h1>```",
            "recipient": "assistant",
        },
        {"role": "assistant", "type": "code", "format": "python", "content": "pass"},
        {"role": "computer", "type": "console", "format": "output", "content": ""},
    ]

    out = convert_to_openai_messages(
        messages,
        function_calling=True,
        vision=False,
        interpreter=_FakeGoDeepSeekInterpreter(),
    )

    calls = [m for m in out if m.get("tool_calls")]
    results = [m for m in out if m.get("role") == "tool"]
    assert len(calls) == 2
    assert len(results) == 2, "one result per tool call"
    assert results[0]["tool_call_id"] == calls[0]["tool_calls"][0]["id"]
    assert results[1]["tool_call_id"] == calls[1]["tool_calls"][0]["id"]
    assert "HTML is being displayed" in results[0]["content"]
    assert "Could not render a preview" in results[0]["content"]
    assert "HTML source shown" in results[0]["content"]
    assert results[1]["content"] == "No output"


def test_repeated_legacy_function_results_are_merged():
    """The legacy function/message shape gets the same one-result guarantee."""
    messages = [
        {"role": "assistant", "type": "code", "format": "html", "content": "<h1>hi</h1>"},
        {"role": "computer", "type": "console", "format": "output", "content": "displayed"},
        {"role": "computer", "type": "console", "format": "output", "content": "source"},
    ]

    out = convert_to_openai_messages(
        messages, function_calling=True, vision=False, interpreter=None
    )

    results = [m for m in out if m.get("role") == "function"]
    assert len(results) == 1
    assert results[0]["content"] == "displayed\nsource"


def test_empty_outputs_collapse_and_do_not_hide_real_output():
    """Multiple empty outputs become one "No output" and never mask real content."""
    from interpreter.core.llm.utils.convert_to_openai_messages import (
        _merge_adjacent_tool_results,
    )

    assert _merge_adjacent_tool_results(
        [
            {"role": "tool", "tool_call_id": "a", "content": "No output"},
            {"role": "tool", "tool_call_id": "a", "content": "No output"},
        ]
    ) == [{"role": "tool", "tool_call_id": "a", "content": "No output"}]

    assert _merge_adjacent_tool_results(
        [
            {"role": "tool", "tool_call_id": "a", "content": "No output"},
            {"role": "tool", "tool_call_id": "a", "content": "real"},
        ]
    ) == [{"role": "tool", "tool_call_id": "a", "content": "real"}]


def test_different_tool_call_ids_are_not_merged():
    """Parallel calls keep their own results."""
    from interpreter.core.llm.utils.convert_to_openai_messages import (
        _merge_adjacent_tool_results,
    )

    merged = _merge_adjacent_tool_results(
        [
            {"role": "tool", "tool_call_id": "a", "content": "one"},
            {"role": "tool", "tool_call_id": "b", "content": "two"},
        ]
    )

    assert len(merged) == 2
