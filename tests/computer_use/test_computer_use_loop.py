import importlib.util
import sys
import types
from pathlib import Path

import pytest

# pyautogui is an [os]-optional dep absent from unit-test CI. Stub before any
# interpreter.computer_use import (tools/__init__.py pulls in ComputerTool).
_stub = types.ModuleType("pyautogui")
sys.modules["pyautogui"] = _stub

_base_path = (
    Path(__file__).resolve().parents[2] / "interpreter/computer_use/tools/base.py"
)
_spec = importlib.util.spec_from_file_location("computer_use_base", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
ToolResult = _base.ToolResult
try:
    from interpreter.computer_use.loop import (
        _make_api_tool_result,
        _maybe_filter_to_n_most_recent_images,
        _maybe_prepend_system_tool_result,
    )
except ImportError as exc:
    pytest.skip(
        f"computer_use.loop helpers unavailable in this environment: {exc}",
        allow_module_level=True,
    )
finally:
    sys.modules.pop("pyautogui", None)


def _tool_messages(image_data):
    """Build a minimal message list with one tool_result block and image entries."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "content": [
                        {"type": "image", "data": data} for data in image_data
                    ],
                }
            ],
        }
    ]


def test_maybe_prepend_system_tool_result():
    """A system note is prepended only when the result carries one."""
    assert (
        _maybe_prepend_system_tool_result(
            ToolResult(system="SYS"), "body"
        )
        == "<system>SYS</system>\nbody"
    )
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
    out = _make_api_tool_result(
        ToolResult(error="boom", system="SYS"), "id2"
    )
    assert out["is_error"] is True
    assert out["content"] == "<system>SYS</system>\nboom"


def test_make_api_tool_result_with_base64_image():
    """A result with a screenshot appends an image block after any text."""
    out = _make_api_tool_result(
        ToolResult(output="o", base64_image="imgdata"), "id3"
    )
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

    _maybe_filter_to_n_most_recent_images(
        messages, images_to_keep=1, min_removal_threshold=1
    )

    remaining = [
        block
        for block in messages[0]["content"][0]["content"]
        if block["type"] == "image"
    ]
    assert remaining == [{"type": "image", "data": "3"}]


def test_maybe_filter_to_n_most_recent_images_batches_by_default_threshold():
    """Default min_removal_threshold=5 removes in chunks once six images exist."""
    messages = _tool_messages(["1", "2", "3", "4", "5", "6"])

    _maybe_filter_to_n_most_recent_images(messages, images_to_keep=1)

    remaining = [
        block
        for block in messages[0]["content"][0]["content"]
        if block["type"] == "image"
    ]
    assert len(remaining) == 1
    assert remaining[0]["data"] == "6"


def test_maybe_filter_to_n_most_recent_images_noop_when_disabled():
    """images_to_keep=None skips filtering entirely."""
    messages = _tool_messages(["1", "2", "3"])

    _maybe_filter_to_n_most_recent_images(messages, images_to_keep=None)

    remaining = [
        block
        for block in messages[0]["content"][0]["content"]
        if block["type"] == "image"
    ]
    assert len(remaining) == 3
