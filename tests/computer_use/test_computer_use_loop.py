import asyncio
import contextlib
import importlib.util
import sys
import types
import unittest.mock
from pathlib import Path

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
from rich.markdown import Markdown
from rich.rule import Rule

# pyautogui is an [os]-optional dep absent from unit-test CI. Stub before any
# interpreter.computer_use import (tools/__init__.py pulls in ComputerTool).
_stub = types.ModuleType("pyautogui")
sys.modules["pyautogui"] = _stub

_base_path = Path(__file__).resolve().parents[2] / "interpreter/computer_use/tools/base.py"
_spec = importlib.util.spec_from_file_location("computer_use_base", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
ToolResult = _base.ToolResult
try:
    import interpreter.computer_use.loop as loop_module
    from interpreter.computer_use.loop import (
        APIProvider,
        ChatCompletionRequest,
        ChatMessage,
        PROVIDER_TO_DEFAULT_MODEL_NAME,
        _make_api_tool_result,
        _maybe_filter_to_n_most_recent_images,
        _maybe_prepend_system_tool_result,
        sampling_loop,
    )
except ImportError as exc:
    pytest.skip(
        f"computer_use.loop helpers unavailable in this environment: {exc}",
        allow_module_level=True,
    )
finally:
    sys.modules.pop("pyautogui", None)


def _run_async(coro):
    """Run a coroutine on a fresh event loop, ignoring the closed-loop default."""
    asyncio.run(coro)


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


def _text_stream(text_parts):
    """Build a single text block streamed in parts, ending with a stop event."""
    events = [
        BetaRawContentBlockStartEvent(
            content_block=BetaTextBlock(text="", type="text"),
            index=0,
            type="content_block_start",
        )
    ]
    for part in text_parts:
        events.append(
            BetaRawContentBlockDeltaEvent(
                delta=BetaTextDelta(text=part, type="text_delta"),
                index=0,
                type="content_block_delta",
            )
        )
    events.append(BetaRawContentBlockStopEvent(index=0, type="content_block_stop"))
    return events


def _tool_stream(name="computer", partial_json='{"action": "screenshot"}'):
    """Build a single tool_use block whose input arrives via input_json_delta events."""
    return [
        BetaRawContentBlockStartEvent(
            content_block=BetaToolUseBlock(id="toolu_1", input={}, name=name, type="tool_use"),
            index=0,
            type="content_block_start",
        ),
        BetaRawContentBlockDeltaEvent(
            delta=BetaInputJSONDelta(partial_json=partial_json, type="input_json_delta"),
            index=0,
            type="content_block_delta",
        ),
        BetaRawContentBlockStopEvent(index=0, type="content_block_stop"),
    ]


def _mock_loop_deps(client, tool_result=None):
    """Return a mocked ToolCollection and the patchers so sampling_loop needs no real client or screen."""
    if tool_result is None:
        tool_result = ToolResult(output="ok")
    tool_collection = unittest.mock.MagicMock()
    tool_collection.to_params.return_value = []
    tool_collection.run = unittest.mock.AsyncMock(return_value=tool_result)
    patchers = (
        unittest.mock.patch.object(loop_module, "ToolCollection", return_value=tool_collection),
        unittest.mock.patch.object(loop_module, "ComputerTool"),
        unittest.mock.patch.object(loop_module, "Anthropic", return_value=client),
    )
    return tool_collection, patchers


def _enter(patchers):
    """Return an ExitStack with every patch active, for `with _enter(...):` use."""
    stack = contextlib.ExitStack()
    for patcher in patchers:
        stack.enter_context(patcher)
    return stack


def _run_sampling_loop(client, messages, **kwargs):
    """Drive sampling_loop with mocked deps and return (yielded chunks, output blocks)."""
    tool_collection, patchers = _mock_loop_deps(client)
    yielded = []
    output_blocks = []
    with _enter(patchers):
        _run_async(
            _collect_sampling_loop(
                sampling_loop(
                    model="test-model",
                    provider=APIProvider.ANTHROPIC,
                    system_prompt_suffix=kwargs.pop("system_prompt_suffix", ""),
                    messages=messages,
                    output_callback=output_blocks.append,
                    tool_output_callback=lambda result, tool_id: None,
                    api_key="test-key",
                    **kwargs,
                ),
                yielded,
            )
        )
    return tool_collection, yielded, output_blocks


async def _collect_sampling_loop(agen, yielded):
    """Drain an async generator, appending every yielded value to the list."""
    async for chunk in agen:
        yielded.append(chunk)


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


def test_sampling_loop_text_only_message_ends_after_one_iteration():
    """A text-only stream yields text chunks then a messages chunk, and stops without calling tools."""
    client = unittest.mock.MagicMock()
    client.beta.messages.create.return_value = iter(_text_stream(["Hello", " world"]))
    messages = []

    tool_collection, yielded, output_blocks = _run_sampling_loop(client, messages)

    assert yielded[0] == {"type": "chunk", "chunk": "Hello"}
    assert yielded[1] == {"type": "chunk", "chunk": " world"}
    assert yielded[-1]["type"] == "messages"
    assert yielded[-1]["messages"] == messages
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    tool_collection.run.assert_not_awaited()
    assert len(output_blocks) == 1


def test_sampling_loop_runs_tool_and_loops_until_text_message():
    """A tool_use block runs the tool, appends a tool_result user message, then loops until a text reply ends it."""
    client = unittest.mock.MagicMock()
    client.beta.messages.create.side_effect = [
        iter(_tool_stream()),
        iter(_text_stream(["all done"])),
    ]
    messages = []

    tool_collection, yielded, output_blocks = _run_sampling_loop(client, messages)

    assert client.beta.messages.create.call_count == 2
    tool_collection.run.assert_awaited_once_with(name="computer", tool_input={"action": "screenshot"})
    assert messages[0]["role"] == "assistant"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["type"] == "tool_result"
    assert yielded[-1]["type"] == "messages"
    assert yielded[-1]["messages"] == messages


def test_sampling_loop_obeys_system_prompt_suffix():
    """system_prompt_suffix is appended to the system prompt sent to the API."""
    client = unittest.mock.MagicMock()
    client.beta.messages.create.return_value = iter(_text_stream(["ok"]))
    messages = []

    _run_sampling_loop(client, messages, system_prompt_suffix="EXTRA")

    _, kwargs = client.beta.messages.create.call_args
    assert kwargs["system"].endswith("EXTRA")


def test_sampling_loop_filters_recent_images_when_configured():
    """only_n_most_recent_images triggers the image-filtering helper each iteration."""
    client = unittest.mock.MagicMock()
    client.beta.messages.create.return_value = iter(_text_stream(["ok"]))
    messages = [{"role": "user", "content": []}]

    with unittest.mock.patch.object(loop_module, "_maybe_filter_to_n_most_recent_images") as mock_filter:
        _run_sampling_loop(client, messages, only_n_most_recent_images=3)

    mock_filter.assert_called_once_with(messages, 3)


def test_sampling_loop_uses_correct_client_for_provider():
    """Each APIProvider selects the matching Anthropic SDK client constructor."""
    for provider, ctor in [
        (APIProvider.ANTHROPIC, "Anthropic"),
        (APIProvider.BEDROCK, "AnthropicBedrock"),
        (APIProvider.VERTEX, "AnthropicVertex"),
    ]:
        client = unittest.mock.MagicMock()
        client.beta.messages.create.return_value = iter(_text_stream(["ok"]))
        messages = []
        tool_collection, patchers = _mock_loop_deps(client)
        with _enter(patchers):
            with unittest.mock.patch.object(loop_module, ctor) as mock_ctor:
                mock_ctor.return_value = client
                _run_async(
                    _collect_sampling_loop(
                        sampling_loop(
                            model="m",
                            provider=provider,
                            system_prompt_suffix="",
                            messages=messages,
                            output_callback=lambda b: None,
                            tool_output_callback=lambda r, t: None,
                            api_key="k",
                        ),
                        [],
                    )
                )
        mock_ctor.assert_called_once()


def test_print_markdown_renders_lines_and_rules():
    """print_markdown prints each non-empty line as Markdown and --- lines as rules."""
    rendered = []

    def fake_print(obj):
        rendered.append(obj)

    with unittest.mock.patch.object(loop_module, "rich_print", side_effect=fake_print):
        loop_module.print_markdown("hello\nworld\n---\n")

    assert isinstance(rendered[0], Markdown)
    assert rendered[0].markup == "hello"
    assert isinstance(rendered[1], Markdown)
    assert rendered[1].markup == "world"
    assert isinstance(rendered[2], Rule)


def test_provider_to_default_model_name_maps_all_providers():
    """Every APIProvider value has a default model configured."""
    for provider in APIProvider:
        assert PROVIDER_TO_DEFAULT_MODEL_NAME[provider]


def test_chat_completion_request_parses_messages():
    """ChatCompletionRequest parses a message list into ChatMessage objects with defaults."""
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
    assert isinstance(req.messages[0], ChatMessage)
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "hi"
    assert req.stream is False
