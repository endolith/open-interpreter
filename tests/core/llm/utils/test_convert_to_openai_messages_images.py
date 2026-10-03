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


def _oversized_jpeg_path(tmp_path, name):
    """A JPEG on disk whose base64 exceeds the 5MB shrink threshold."""
    import os

    from PIL import Image

    img = Image.frombytes("RGB", (2200, 2200), os.urandom(2200 * 2200 * 3))
    p = tmp_path / name
    img.save(str(p), format="JPEG", quality=100)
    return str(p)


@pytest.mark.parametrize("name", ["photo.jpeg", "photo.jpg", "photo.JPG"])
def test_shrinking_a_large_jpeg_does_not_raise_on_the_extension(interpreter, tmp_path, name):
    """A .jpg/.JPG over 5MB shrinks instead of crashing with KeyError from PIL.

    The extension was passed to Image.save verbatim, but PIL registers the save
    format as "JPEG" — "JPG"/"jpg" raise KeyError. That propagated out of
    Llm.run and, via the bare except in the terminal, killed the CLI. The
    extension is now mapped through PIL's own registry, and the emitted data
    URI mimetype is normalised (jpg -> jpeg) to match.
    """
    path = _oversized_jpeg_path(tmp_path, name)
    messages = [{"role": "user", "type": "image", "format": "path", "content": path}]

    result = convert_to_openai_messages(
        messages, vision=True, shrink_images=True, interpreter=interpreter
    )

    url = result[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    # actually shrunk below the 5 MB budget
    assert len(url) / 1024 / 1024 < 5
