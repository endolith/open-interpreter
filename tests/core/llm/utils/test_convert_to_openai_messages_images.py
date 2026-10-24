import base64
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


class _StubImage:
    """A PIL Image stand-in whose resize never actually shrinks anything.

    The real shrink loop recomputes the encoded size after every pass and breaks
    once it is under the limit. A stub whose saved payload never gets smaller is
    the only way to reach the loop's exhaustion branch, which is unreachable with
    a real image: the scale factor is `(4.9 / size_mb) ** 0.5` and area scales as
    the square of that, so one pass from 10x oversize lands near 4.9 MB and always
    succeeds.
    """

    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        self.resizes = 0
        # Deliberately different from the input bytes, so the emitted URL's
        # payload can be checked against what the stub actually saved rather than
        # only for having a data: prefix.
        self.saved_bytes = b"\x89PNG\r\n\x1a\n" + b"1" * (6 * 1024 * 1024)

    def resize(self, size):
        self.resizes += 1
        return self

    def save(self, buffered, format=None):  # noqa: A002 - PIL's parameter name
        # Always re-encodes to something still over the limit, so no pass can ever
        # satisfy the break condition and the loop runs to exhaustion.
        buffered.write(self.saved_bytes)


def test_shrink_gives_up_after_ten_passes_and_still_sends_the_image(
    interpreter, monkeypatch, capsys
):
    """An image that will not shrink is sent anyway, after ten attempts.

    The loop is bounded, and its `else` branch says so and falls through. The
    invariant worth protecting is that the image is *still delivered*: dropping it
    would silently remove the screenshot the model was asked to look at, and the
    `for`/`else` is the only thing standing between a stubborn image and a blank
    payload.
    """
    stub = _StubImage()

    monkeypatch.setattr(
        "interpreter.core.llm.utils.convert_to_openai_messages.Image.open",
        lambda *a, **k: stub,
    )

    original_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * (6 * 1024 * 1024)).decode()
    messages = [
        {
            "role": "computer",
            "type": "image",
            "format": "base64.png",
            "content": original_b64,
        }
    ]

    result = convert_to_openai_messages(messages, interpreter=interpreter, vision=True)

    url = result[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # The payload is what the stub saved. Checking only the prefix would pass for
    # an empty body, or for the original input echoed back unchanged.
    assert base64.b64decode(url.split(",", 1)[1]) == stub.saved_bytes
    assert stub.resizes == 10, "the retry budget is ten passes"
    assert "Attempted to shrink the image but failed" in capsys.readouterr().out


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


def test_unknown_extension_labels_bytes_by_detected_format(interpreter):
    """A JPEG with an unrecognized extension is served as image/jpeg, not png.

    With an unknown extension the bytes pass through unmodified (below the
    shrink threshold), so the data URI mimetype must describe the actual
    bytes. Falling back to PNG mislabeled JPEG bytes as data:image/png.
    The real format is now detected from the image header (CodeRabbit
    review of #347).
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="JPEG")
    raw = buf.getvalue()
    messages = [
        {
            "role": "user",
            "type": "image",
            "format": "base64.bin",
            "content": base64.b64encode(raw).decode(),
        }
    ]

    result = convert_to_openai_messages(
        messages, vision=True, shrink_images=True, interpreter=interpreter
    )

    url = result[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == raw


def test_garbage_bytes_with_unknown_extension_fall_back_to_png(interpreter):
    """Undetectable bytes under an unknown extension stay PNG-labelled.

    When PIL cannot even open the payload, there is no detected format, so
    the PNG fallback applies as before instead of crashing the conversion.
    """
    messages = [
        {
            "role": "user",
            "type": "image",
            "format": "base64.bin",
            "content": base64.b64encode(b"not an image at all").decode(),
        }
    ]

    result = convert_to_openai_messages(
        messages, vision=True, shrink_images=False, interpreter=interpreter
    )

    url = result[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


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


def test_image_unpngable_mode_falls_back_to_rgb(interpreter):
    """A mode PNG cannot store (CMYK TIFF) is converted to RGB first (issue #377)."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("CMYK", (8, 8)).save(buf, format="TIFF")
    raw = base64.b64encode(buf.getvalue()).decode()
    messages = [
        {"role": "user", "type": "image", "format": "base64.tiff", "content": raw}
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


def test_image_undecodable_bytes_keep_declared_extension(interpreter):
    """Bytes the probe can name but neither save path can decode are passed through (issue #377).

    A truncated BMP still reports format BMP, so it reaches the re-encode
    path — but both the direct PNG save and the RGB fallback raise while
    decoding. The original payload and declared extension must be preserved
    instead of letting the exception escape.
    """
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="BMP")
    raw = base64.b64encode(buf.getvalue()[:60]).decode()
    messages = [
        {"role": "user", "type": "image", "format": "base64.bmp", "content": raw}
    ]
    url = _image_url(
        convert_to_openai_messages(
            messages, vision=True, shrink_images=False, interpreter=interpreter
        )
    )
    assert url == f"data:image/bmp;base64,{raw}"


def test_image_mpo_is_treated_as_jpeg(interpreter):
    """Pillow reports MPO, but it is a JPEG container; keep the JPEG bytes."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    first = Image.new("RGB", (8, 8), (200, 30, 30))
    second = Image.new("RGB", (8, 8), (30, 200, 30))
    first.save(buf, format="MPO", save_all=True, append_images=[second])
    raw = base64.b64encode(buf.getvalue()).decode()
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
