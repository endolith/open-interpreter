from types import SimpleNamespace

import pytest

from interpreter.core.llm.utils.convert_to_openai_messages import (
    convert_to_openai_messages,
)


@pytest.fixture
def interpreter():
    return SimpleNamespace(
        user_message_template="{content}",
        always_apply_user_message_template=False,
        code_output_template="Output:\n{content}",
        empty_code_output_template="(no output)",
        code_output_sender="assistant",
        debug=False,
    )


def test_image_base64_with_vision(interpreter):
    """Base64 PNG images are converted to OpenAI image_url parts when vision is enabled."""
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
    result = convert_to_openai_messages(
        messages, vision=True, shrink_images=False, interpreter=interpreter
    )
    assert result[0]["content"][0]["type"] == "image_url"
    assert "data:image/png;base64," in result[0]["content"][0]["image_url"]["url"]


def _encode_image(pil_format):
    """Build small real image bytes in the given PIL format."""
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", (8, 8), color=(200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format=pil_format)
    return base64.b64encode(buf.getvalue()).decode()


def _image_url(result):
    return result[0]["content"][0]["image_url"]["url"]


def test_image_mime_sniffed_from_bytes_not_extension(interpreter):
    """JPEG bytes mislabelled base64.png go out as image/jpeg, unchanged (issue #377)."""
    import base64

    raw = _encode_image("JPEG")
    messages = [
        {"role": "user", "type": "image", "format": "base64.png", "content": raw}
    ]
    url = _image_url(
        convert_to_openai_messages(
            messages, vision=True, shrink_images=False, interpreter=interpreter
        )
    )
    assert url.startswith("data:image/jpeg;base64,")
    assert url.split(",", 1)[1] == raw


def test_image_unsupported_format_reencoded_to_png(interpreter):
    """BMP bytes are re-encoded to PNG since providers lack BMP support (issue #377)."""
    import base64

    messages = [
        {
            "role": "user",
            "type": "image",
            "format": "base64.bmp",
            "content": _encode_image("BMP"),
        }
    ]
    url = _image_url(
        convert_to_openai_messages(
            messages, vision=True, shrink_images=False, interpreter=interpreter
        )
    )
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


def test_image_supported_format_passes_through_untouched(interpreter):
    """PNG bytes keep their bytes and MIME (issue #377)."""
    raw = _encode_image("PNG")
    messages = [
        {"role": "user", "type": "image", "format": "base64.png", "content": raw}
    ]
    url = _image_url(
        convert_to_openai_messages(
            messages, vision=True, shrink_images=False, interpreter=interpreter
        )
    )
    assert url == f"data:image/png;base64,{raw}"


def test_image_path_extension_ignored_in_favor_of_bytes(interpreter, tmp_path):
    """A JPEG saved under a .png path is still labelled image/jpeg (issue #377)."""
    import base64

    target = tmp_path / "photo.png"
    target.write_bytes(base64.b64decode(_encode_image("JPEG")))
    messages = [
        {"role": "user", "type": "image", "format": "path", "content": str(target)}
    ]
    url = _image_url(
        convert_to_openai_messages(
            messages, vision=True, shrink_images=False, interpreter=interpreter
        )
    )
    assert url.startswith("data:image/jpeg;base64,")
