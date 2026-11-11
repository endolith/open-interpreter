"""Tests for binary websocket input reaching interpreter.messages (#579).

Binary frames are appended to the open message's content, but they used to be
stored as raw ``bytes``. ``convert_to_openai_messages`` cannot serialise that:
its user-message template calls ``str.replace`` (which rejects bytes), and its
image branch base64-decodes text that it assumes is already encoded. These
tests drive ``AsyncInterpreter.input`` through a completed start/bytes/end
exchange -- the moment ``respond`` would convert the messages -- and assert
both the stored representation and the converted request.
"""

import asyncio
import base64
import io

import pytest
from PIL import Image

from interpreter.core.async_core import AsyncInterpreter
from interpreter.core.llm.utils.convert_to_openai_messages import (
    convert_to_openai_messages,
)


def _png_bytes():
    """A small real PNG so the converter's image branch can decode it."""
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buffer, "PNG")
    return buffer.getvalue()


def _deliver(interpreter, chunks):
    """Send each chunk then close the message, with respond() stubbed out.

    respond() starts a worker thread that calls the LLM; the assertions are
    about the accumulated messages, so it must not run.
    """

    async def deliver():
        for chunk in chunks:
            await interpreter.input(chunk)
        await interpreter.input({"end": True})

    interpreter.respond = lambda run_code=None: None
    asyncio.run(deliver())


def test_binary_text_frames_serialise_for_the_request_converter():
    """UTF-8 binary frames decode to text the converter's template can use.

    Storing bytes made ``str.replace`` raise TypeError and killed the response
    when respond() converted the completed message.
    """
    interpreter = AsyncInterpreter()
    _deliver(
        interpreter,
        [
            {"role": "user", "type": "message", "start": True},
            b"hello ",
            b"world",
        ],
    )
    assert interpreter.messages[-1]["content"] == "hello world"
    converted = convert_to_openai_messages(
        interpreter.messages, interpreter=interpreter
    )
    assert converted[-1] == {"role": "user", "content": "hello world"}


def test_text_after_binary_keeps_wire_order():
    """A text chunk flushes buffered binary frames first, so nothing is reordered."""
    interpreter = AsyncInterpreter()
    _deliver(
        interpreter,
        [
            {"role": "user", "type": "message", "start": True},
            b"one",
            {"role": "user", "type": "message", "content": " two"},
        ],
    )
    assert interpreter.messages[-1]["content"] == "one two"


def test_binary_image_frames_become_base64_text_the_converter_accepts():
    """Image payloads are base64-encoded at ingest; the converter emits a data URI."""
    payload = _png_bytes()
    interpreter = AsyncInterpreter()
    chunks = [{"role": "user", "type": "image", "format": "png.base64", "start": True}]
    # Split the payload across frames so a UTF-8 or base64 bug on frame
    # boundaries cannot hide.
    chunks += [payload[i : i + 7] for i in range(0, len(payload), 7)]
    _deliver(interpreter, chunks)
    assert interpreter.messages[-1]["content"] == base64.b64encode(payload).decode(
        "ascii"
    )
    converted = convert_to_openai_messages(
        interpreter.messages, interpreter=interpreter, vision=True
    )
    content = converted[-1]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/")
    assert "base64" in content[0]["image_url"]["url"]


def test_binary_frames_for_non_text_messages_are_rejected():
    """Only text and image messages absorb binary frames; code is refused."""
    interpreter = AsyncInterpreter()

    async def deliver():
        await interpreter.input(
            {"role": "assistant", "type": "code", "format": "python", "start": True}
        )
        with pytest.raises(
            ValueError, match="not supported for message type 'code'"
        ):
            await interpreter.input(b"print(1)")

    asyncio.run(deliver())


def test_binary_text_must_be_utf8():
    """A text message rejects bytes that are not UTF-8 instead of storing garbage."""
    interpreter = AsyncInterpreter()

    async def deliver():
        await interpreter.input({"role": "user", "type": "message", "start": True})
        await interpreter.input(b"\xff\xfe")
        with pytest.raises(ValueError, match="must be UTF-8"):
            await interpreter.input({"end": True})

    asyncio.run(deliver())


def test_binary_frames_buffer_until_the_message_closes():
    """Frames append to a bytearray; content materialises once, not per frame."""
    interpreter = AsyncInterpreter()

    async def deliver():
        await interpreter.input({"role": "user", "type": "message", "start": True})
        await interpreter.input(b"abc")
        assert interpreter.messages[-1]["content"] == ""
        assert bytes(interpreter._binary_buffer) == b"abc"
        await interpreter.input({"end": True})

    interpreter.respond = lambda run_code=None: None
    asyncio.run(deliver())
    assert interpreter.messages[-1]["content"] == "abc"
