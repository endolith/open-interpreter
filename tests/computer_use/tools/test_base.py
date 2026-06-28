import importlib.util
from pathlib import Path
from unittest import mock

import pytest

_base_path = (
    Path(__file__).resolve().parents[3] / "interpreter/computer_use/tools/base.py"
)
_spec = importlib.util.spec_from_file_location("computer_use_base", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

ToolResult = _base.ToolResult


def test_tool_result_add_concatenates_strings():
    combined = ToolResult(output="hello") + ToolResult(output=" world")
    assert combined.output == "hello world"


def test_tool_result_add_conflicting_images_raises():
    with pytest.raises(ValueError, match="Cannot combine tool results"):
        _ = ToolResult(base64_image="aaa") + ToolResult(base64_image="bbb")
