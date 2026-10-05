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

    def resize(self, size):
        self.resizes += 1
        return self

    def save(self, buffered, format=None):  # noqa: A002 - PIL's parameter name
        # Always re-encodes to something still over the limit, so no pass can ever
        # satisfy the break condition and the loop runs to exhaustion.
        buffered.write(b"\x89PNG\r\n\x1a\n" + b"0" * (6 * 1024 * 1024))


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
    assert stub.resizes == 10, "the retry budget is ten passes"
    assert "Attempted to shrink the image but failed" in capsys.readouterr().out
