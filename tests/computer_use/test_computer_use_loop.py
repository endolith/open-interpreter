import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
from anthropic.types.beta import (
    BetaInputJSONDelta,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockStopEvent,
    BetaTextBlock,
    BetaTextDelta,
    BetaToolUseBlock,
)

# pyautogui is an [os]-optional dep absent from unit-test CI. Stub before any
# interpreter.computer_use import (tools/__init__.py pulls in ComputerTool),
# restoring any previously installed module afterwards.
_stub = types.ModuleType("pyautogui")
_stub_previous = sys.modules.get("pyautogui")
sys.modules["pyautogui"] = _stub

_base_path = Path(__file__).resolve().parents[2] / "interpreter/computer_use/tools/base.py"
_spec = importlib.util.spec_from_file_location("computer_use_base", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
ToolResult = _base.ToolResult
try:
    from interpreter.computer_use.loop import (
        APIProvider,
        _make_api_tool_result,
        _maybe_filter_to_n_most_recent_images,
        _maybe_prepend_system_tool_result,
        sampling_loop,
    )
    from interpreter.computer_use.tools.base import (
        ToolError as RealToolError,
    )
except ImportError as exc:
    pytest.skip(
        f"computer_use.loop helpers unavailable in this environment: {exc}",
        allow_module_level=True,
    )
finally:
    if _stub_previous is None:
        sys.modules.pop("pyautogui", None)
    else:
        sys.modules["pyautogui"] = _stub_previous


def _tool_messages(image_data):
    """Build a minimal message list with one tool_result block and image entries."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "content": [{"type": "image", "data": data} for data in image_data],
                }
            ],
        }
    ]


def test_maybe_prepend_system_tool_result():
    """A system note is prepended only when the result carries one."""
    assert _maybe_prepend_system_tool_result(ToolResult(system="SYS"), "body") == "<system>SYS</system>\nbody"
    assert _maybe_prepend_system_tool_result(ToolResult(system=""), "body") == "body"


def test_make_api_tool_result_with_output_only():
    """A plain tool result becomes a single text block."""
    out = _make_api_tool_result(ToolResult(output="hello"), "id1")
    assert out["type"] == "tool_result"
    assert out["tool_use_id"] == "id1"
    assert out["is_error"] is False
    assert out["content"] == [{"type": "text", "text": "hello"}]


def test_make_api_tool_result_with_error_prepends_system():
    """An error result is flagged and stores prepended system text as a plain string."""
    out = _make_api_tool_result(ToolResult(error="boom", system="SYS"), "id2")
    assert out["is_error"] is True
    assert out["content"] == "<system>SYS</system>\nboom"


def test_make_api_tool_result_with_base64_image():
    """A result with a screenshot appends an image block after any text."""
    out = _make_api_tool_result(ToolResult(output="o", base64_image="imgdata"), "id3")
    assert out["content"][0] == {"type": "text", "text": "o"}
    assert {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "imgdata",
        },
    } in out["content"]


def test_maybe_filter_to_n_most_recent_images_keeps_last_n():
    """With min_removal_threshold=1, only the newest screenshots remain."""
    messages = _tool_messages(["1", "2", "3"])

    _maybe_filter_to_n_most_recent_images(messages, images_to_keep=1, min_removal_threshold=1)

    remaining = [block for block in messages[0]["content"][0]["content"] if block["type"] == "image"]
    assert remaining == [{"type": "image", "data": "3"}]


def test_maybe_filter_to_n_most_recent_images_batches_by_default_threshold():
    """Default min_removal_threshold=5 removes in chunks once six images exist."""
    messages = _tool_messages(["1", "2", "3", "4", "5", "6"])

    _maybe_filter_to_n_most_recent_images(messages, images_to_keep=1)

    remaining = [block for block in messages[0]["content"][0]["content"] if block["type"] == "image"]
    assert len(remaining) == 1
    assert remaining[0]["data"] == "6"


def test_maybe_filter_to_n_most_recent_images_noop_when_disabled():
    """images_to_keep=None skips filtering entirely."""
    messages = _tool_messages(["1", "2", "3"])

    _maybe_filter_to_n_most_recent_images(messages, images_to_keep=None)

    remaining = [block for block in messages[0]["content"][0]["content"] if block["type"] == "image"]
    assert len(remaining) == 3


def _tool_use_stream(action_json, tool_id="toolu_1"):
    """Build the Anthropic stream events for a single tool-use block."""
    return [
        BetaRawContentBlockStartEvent(
            content_block=BetaToolUseBlock(id=tool_id, input={}, name="computer", type="tool_use"),
            index=0,
            type="content_block_start",
        ),
        BetaRawContentBlockDeltaEvent(
            delta=BetaInputJSONDelta(partial_json=action_json, type="input_json_delta"),
            index=0,
            type="content_block_delta",
        ),
        BetaRawContentBlockStopEvent(index=0, type="content_block_stop"),
    ]


def _text_stream(text, index=0):
    """Build the Anthropic stream events for a single text block."""
    return [
        BetaRawContentBlockStartEvent(
            content_block=BetaTextBlock(text="", type="text"),
            index=index,
            type="content_block_start",
        ),
        BetaRawContentBlockDeltaEvent(
            delta=BetaTextDelta(text=text, type="text_delta"),
            index=index,
            type="content_block_delta",
        ),
        BetaRawContentBlockStopEvent(index=index, type="content_block_stop"),
    ]


class _FakeComputerTool:
    """Test double for ComputerTool; subclasses decide what __call__ does."""

    def to_params(self):
        return {
            "name": "computer",
            "input_schema": {"type": "object", "properties": {}},
        }

    async def __call__(self, **kwargs):
        raise NotImplementedError


class _ScreenshotComputerTool(_FakeComputerTool):
    async def __call__(self, **kwargs):
        return ToolResult(output="fake screenshot taken")


class _FailingComputerTool(_FakeComputerTool):
    async def __call__(self, **kwargs):
        raise RealToolError("could not screenshot")


def _fake_anthropic(streams):
    """Return a factory for clients that replay the given stream events in order."""

    stream_iter = iter(streams)

    class _FakeBetaMessages:
        def create(self, **kwargs):
            return next(stream_iter)

    class _FakeBeta:
        messages = _FakeBetaMessages()

    class _FakeAnthropic:
        def __init__(self, api_key=None, **kwargs):
            self.beta = _FakeBeta()

    return _FakeAnthropic


def _run_sampling_loop(streams, *, tool_cls=_ScreenshotComputerTool):
    """Drive sampling_loop with fake streams and a fake tool, capturing callbacks."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "take a screenshot"}]}]
    output_blocks = []
    tool_outputs = []
    with (
        mock.patch("interpreter.computer_use.loop.Anthropic", _fake_anthropic(streams)),
        mock.patch("interpreter.computer_use.loop.ComputerTool", tool_cls),
    ):

        async def _consume():
            chunks = []
            async for chunk in sampling_loop(
                model="fake-model",
                provider=APIProvider.ANTHROPIC,
                system_prompt_suffix="",
                messages=messages,
                output_callback=output_blocks.append,
                tool_output_callback=lambda result, tool_id: tool_outputs.append((result, tool_id)),
                api_key="fake-key",
                only_n_most_recent_images=10,
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_consume())
    return messages, chunks, output_blocks, tool_outputs


def test_sampling_loop_runs_tool_then_finishes_with_text():
    """sampling_loop executes a tool call and finishes once the model replies with text only."""
    messages, chunks, output_blocks, tool_outputs = _run_sampling_loop(
        [
            _tool_use_stream('{"action": "screenshot"}'),
            _text_stream("All done."),
        ]
    )

    assert messages[1]["role"] == "assistant"
    tool_use = messages[1]["content"][0]
    assert tool_use.type == "tool_use"
    assert tool_use.input == {"action": "screenshot"}

    tool_result = messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["is_error"] is False
    assert tool_result["content"] == [{"type": "text", "text": "fake screenshot taken"}]

    assert messages[3]["role"] == "assistant"
    assert messages[3]["content"][0].text == "All done."

    assert [block.type for block in output_blocks] == ["tool_use", "text"]
    assert len(tool_outputs) == 1
    assert tool_outputs[0][0].output == "fake screenshot taken"

    assert chunks[-1]["type"] == "messages"
    assert [c["chunk"] for c in chunks if c["type"] == "chunk"] == [
        "All done.",
        "\n",
    ]


def test_sampling_loop_finishes_immediately_without_tool_calls():
    """A text-only response ends the loop in one iteration without running any tool."""
    messages, chunks, output_blocks, tool_outputs = _run_sampling_loop([_text_stream("Hello")])

    assert len(messages) == 2
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0].text == "Hello"
    assert tool_outputs == []
    assert output_blocks == [messages[1]["content"][0]]

    assert chunks[-1]["type"] == "messages"
    assert [c["chunk"] for c in chunks if c["type"] == "chunk"] == ["Hello", "\n"]


def test_sampling_loop_tool_error_becomes_failure_and_loop_continues():
    """A ToolError raised by a tool becomes an error result and the loop keeps going."""
    messages, chunks, output_blocks, tool_outputs = _run_sampling_loop(
        [
            _tool_use_stream('{"action": "click"}'),
            _text_stream("recovered"),
        ],
        tool_cls=_FailingComputerTool,
    )

    tool_result = messages[2]["content"][0]
    assert tool_result["is_error"] is True
    assert "could not screenshot" in tool_result["content"]

    assert messages[3]["content"][0].text == "recovered"
    assert len(tool_outputs) == 1
    assert tool_outputs[0][0].error == "could not screenshot"
    assert chunks[-1]["type"] == "messages"
