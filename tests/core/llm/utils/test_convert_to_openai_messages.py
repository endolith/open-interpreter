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
        convert_to_openai_messages(messages, vision=False, interpreter=interpreter)
        == []
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
    with pytest.raises(Exception, match="format"):
        convert_to_openai_messages(
            [{"role": "user", "type": "image", "content": "data"}],
            vision=True,
            interpreter=interpreter,
        )


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
    messages = [{"role": "user", "type": "image", "format": "base64", "content": png}]
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
    messages = [{"role": "assistant", "type": "message", "content": "  Hello world  "}]
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


import base64
import io
import sys


def _png_bytes():
    """A small valid 2x2 PNG, for tests that need a decodable image."""
    import io

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
    url = result[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_path_image_extension_from_filename(interpreter, tmp_path):
    """For image paths, the data URL's extension comes from the file name and the base64 payload is the file's bytes."""
    path = tmp_path / "pic.v2.png"
    path.write_bytes(_png_bytes())
    messages = [
        {"role": "user", "type": "image", "format": "path", "content": str(path)}
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter, vision=True)
    url = result[0]["content"][0]["image_url"]["url"]
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
    import sys as _sys

    from PIL import Image

    path, width = _oversized_png_path(tmp_path)
    original_b64 = base64.b64encode(path.read_bytes()).decode()
    original_content = "data:image/png;base64," + original_b64

    messages = [
        {"role": "computer", "type": "image", "format": "path", "content": str(path)}
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter, vision=True)

    url = result[0]["content"][0]["image_url"]["url"]
    # Shrinking happened at all.
    assert len(url) < len(original_content)
    # The scale factor matches (4.9 / size_mb) ** 0.5 applied once.
    size_mb = _sys.getsizeof(original_content) / (1024 * 1024)
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
    assert result[1]["content"][0]["type"] == "image_url"
