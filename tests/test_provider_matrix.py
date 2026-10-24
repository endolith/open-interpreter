"""Live provider matrix checks, driven through OI's own message conversion.

These run in the scheduled or path-filtered provider matrix, not on every pull
request: each one is cheap, but they multiply across models. Building the
messages with ``convert_to_openai_messages`` keeps the test on OI's pipeline, so
a provider change and an OI conversion regression are both caught here.
"""

import base64
import os

import litellm
import pytest

from interpreter.core.llm.utils.convert_to_openai_messages import (
    convert_to_openai_messages,
)

pytestmark = [pytest.mark.integration, pytest.mark.provider_matrix]

MODEL = os.environ.get("OI_INTEGRATION_MODEL", "openrouter/openai/gpt-4o-mini")

# 8x8 solid red PNG.
RED_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAFElEQVR4nGO8IyLCgA0wYRUdtBIA0OgBFI+iBHoAAAAASUVORK5CYII="
)


class _Interpreter:
    """Minimal interpreter for the conversion path used by these tests."""

    user_message_template = "{content}"
    always_apply_user_message_template = False
    debug = False


def _enabled(name: str) -> bool:
    """Read a capability the matrix entry declares for this model."""
    return os.environ.get(name, "1") == "1"


def _convert(messages, **kwargs):
    """Run messages through OI's conversion, not a hand-built wire shape."""
    return convert_to_openai_messages(
        messages, interpreter=_Interpreter(), **kwargs
    )


def _has_image_part(messages) -> bool:
    """Whether the converted messages carry an image_url content part."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
        ):
            return True
    return False


def test_image_input_is_accepted():
    """OI's image conversion must survive the provider round trip and name red.

    OI's own vision-support detection is unit-tested in
    ``tests/core/llm/test_vision_detection.py``; this drives the same image
    message through ``convert_to_openai_messages`` and the real provider.
    """
    if not _enabled("OI_PROVIDER_VISION"):
        pytest.skip("provider matrix entry declares no vision")
    messages = _convert(
        [
            {
                "role": "user",
                "type": "message",
                "content": "What colour is this image? Answer with one word.",
            },
            {
                "role": "user",
                "type": "image",
                "format": "base64.png",
                "content": RED_PNG,
            },
        ],
        vision=True,
        shrink_images=False,
    )
    assert _has_image_part(messages)
    reply = litellm.completion(model=MODEL, messages=messages, max_tokens=512)
    assert "red" in (reply.choices[0].message.content or "").lower()


def test_tool_call_round_trip():
    """OI's converted prompt must still get a tool call back."""
    if not _enabled("OI_PROVIDER_TOOLS"):
        pytest.skip("provider matrix entry declares no tool calling")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            },
        }
    ]
    messages = _convert(
        [
            {
                "role": "user",
                "type": "message",
                "content": "Use the add tool to add 2 and 2.",
            }
        ]
    )
    reply = litellm.completion(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="required",
        max_tokens=200,
    )
    assert reply.choices[0].message.tool_calls


def test_legacy_function_role_is_accepted():
    """OI still converts executed output to role=function; the provider must accept it."""
    messages = _convert(
        [
            {
                "role": "user",
                "type": "message",
                "content": "What is the weather in Paris?",
            },
            {
                "role": "assistant",
                "type": "code",
                "format": "python",
                "content": "print('sunny')",
            },
            {
                "role": "computer",
                "type": "console",
                "format": "output",
                "content": "sunny, 22C",
            },
            {
                "role": "user",
                "type": "message",
                "content": "In one word, what is the weather?",
            },
        ]
    )
    assert any(message.get("role") == "function" for message in messages)
    reply = litellm.completion(model=MODEL, messages=messages, max_tokens=512)
    assert (reply.choices[0].message.content or "").strip()


def test_final_content_survives():
    """A reasoning model must still return final content, not reasoning only."""
    messages = _convert(
        [
            {
                "role": "user",
                "type": "message",
                "content": "What is 2+2? Answer with the digit.",
            }
        ]
    )
    reply = litellm.completion(model=MODEL, messages=messages, max_tokens=2048)
    assert "4" in (reply.choices[0].message.content or "")
