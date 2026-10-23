"""Live provider quirks that only a real API call reveals.

These run in the scheduled or path-filtered provider matrix, not on every pull
request: each one is cheap, but they multiply across models.
"""

import os

import litellm
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.quirk]

MODEL = os.environ.get("OI_INTEGRATION_MODEL", "gpt-4o-mini")

# 8x8 solid red PNG.
RED_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAFElEQVR4nGO8IyLCgA0wYRUdtBIA0OgBFI+iBHoAAAAASUVORK5CYII="
)


def _bare_model() -> str:
    """Return the model id litellm looks up, without the router prefix."""
    return MODEL.split("/", 1)[1] if "/" in MODEL else MODEL


def test_image_input_is_accepted():
    """The provider must accept an image and name its colour.

    OI's own vision-support detection is unit-tested in
    ``tests/core/llm/test_vision_detection.py`` and the end-to-end OI path is
    covered by the integration suite; this checks the provider contract.
    """
    if not litellm.supports_vision(_bare_model()):
        pytest.skip(f"{MODEL} is not vision-capable")
    reply = litellm.completion(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What colour is this image? Answer with one word.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{RED_PNG}"},
                    },
                ],
            }
        ],
        max_tokens=20,
    )
    assert "red" in (reply.choices[0].message.content or "").lower()


def test_tool_call_round_trip():
    """The provider must return a tool call when the prompt asks for one."""
    if not litellm.supports_function_calling(_bare_model()):
        pytest.skip(f"{MODEL} does not advertise tool calling")
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
    reply = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Use the add tool to add 2 and 2."}],
        tools=tools,
        tool_choice="required",
        max_tokens=200,
    )
    assert reply.choices[0].message.tool_calls


def test_legacy_function_role_is_accepted():
    """OI still sends executed output as role=function; the provider must accept it."""
    reply = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
            },
            {"role": "function", "name": "get_weather", "content": "sunny, 22C"},
            {"role": "user", "content": "In one word, what is the weather?"},
        ],
        max_tokens=512,
    )
    assert (reply.choices[0].message.content or "").strip()


def test_final_content_survives_reasoning_models():
    """A reasoning model must still return final content, not reasoning only."""
    reply = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2+2? Answer with the digit."}],
        max_tokens=512,
    )
    assert "4" in (reply.choices[0].message.content or "")
