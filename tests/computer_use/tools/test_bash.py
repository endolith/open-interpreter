import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

_base = Path(__file__).resolve().parents[3] / "interpreter/computer_use/tools"

for mod_name, filename in [
    ("interpreter.computer_use.tools.base", "base.py"),
    ("interpreter.computer_use.tools.bash", "bash.py"),
]:
    spec = importlib.util.spec_from_file_location(mod_name, _base / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

_bash = sys.modules["interpreter.computer_use.tools.bash"]
BashTool = _bash.BashTool
ToolError = sys.modules["interpreter.computer_use.tools.base"].ToolError


async def _fake_bash_session_start(self):
    """Avoid spawning a real shell (os.setsid is Unix-only; breaks on Windows)."""
    self._started = True
    self._process = mock.Mock(returncode=None)


@pytest.fixture
def bash_without_subprocess():
    with mock.patch.object(_bash._BashSession, "start", _fake_bash_session_start):
        yield


def test_bash_tool_restart(bash_without_subprocess):
    """BashTool with restart=True reports a successful restart in the system message without running a command."""
    tool = BashTool()
    result = asyncio.run(tool(restart=True))
    assert "restarted" in result.system


def test_bash_tool_no_command_raises(bash_without_subprocess):
    """BashTool raises ToolError when invoked without a command, since there is nothing to execute."""
    tool = BashTool()
    with pytest.raises(ToolError, match="no command"):
        asyncio.run(tool())
