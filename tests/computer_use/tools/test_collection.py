import asyncio
import importlib.util
import sys
from pathlib import Path

_base = Path(__file__).resolve().parents[3] / "interpreter/computer_use/tools"

for mod_name, rel in [
    ("interpreter.computer_use.tools.base", "base.py"),
    ("interpreter.computer_use.tools.collection", "collection.py"),
]:
    spec = importlib.util.spec_from_file_location(mod_name, _base / rel.split("/")[-1])
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

ToolCollection = sys.modules["interpreter.computer_use.tools.collection"].ToolCollection
ToolError = sys.modules["interpreter.computer_use.tools.base"].ToolError
ToolFailure = sys.modules["interpreter.computer_use.tools.base"].ToolFailure
ToolResult = sys.modules["interpreter.computer_use.tools.base"].ToolResult


class DummyTool:
    def to_params(self):
        return {"name": "demo", "type": "custom"}

    async def __call__(self, **kwargs):
        return ToolResult(output=f"ok:{kwargs.get('x')}")


def test_tool_collection_runs_registered_tool():
    """ToolCollection.run dispatches to a registered tool by name and returns its ToolResult."""
    collection = ToolCollection(DummyTool())
    result = asyncio.run(collection.run(name="demo", tool_input={"x": 1}))
    assert result.output == "ok:1"


def test_tool_collection_unknown_tool_returns_failure():
    """ToolCollection.run returns ToolFailure instead of raising when the requested tool name is not registered."""
    collection = ToolCollection(DummyTool())
    result = asyncio.run(collection.run(name="missing", tool_input={}))
    assert isinstance(result, ToolFailure)


def test_tool_collection_tool_error_becomes_failure():
    """ToolCollection.run catches ToolError from a tool and wraps it in ToolFailure so callers get a structured error."""
    class BadTool(DummyTool):
        async def __call__(self, **kwargs):
            raise ToolError("boom")

    collection = ToolCollection(BadTool())
    result = asyncio.run(collection.run(name="demo", tool_input={}))
    assert result.error == "boom"
