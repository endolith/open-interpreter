import asyncio
import builtins
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
ToolResult = sys.modules["interpreter.computer_use.tools.base"].ToolResult


class _FakeStreamWriter:
    """Stands in for asyncio's StreamWriter so _BashSession.run can write and drain without a real pipe."""

    def __init__(self):
        self.written = bytearray()

    def write(self, data):
        self.written.extend(data)

    async def drain(self):
        pass


class _FakeProcess:
    """Stands in for an asyncio subprocess, exposing the stream buffers _BashSession.run reads from."""

    def __init__(self, stdout=b"", stderr=b"", returncode=None):
        self.stdin = _FakeStreamWriter()
        self.stdout = mock.Mock()
        self.stdout._buffer = bytearray(stdout)
        self.stderr = mock.Mock()
        self.stderr._buffer = bytearray(stderr)
        self.returncode = returncode


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


def test_bash_tool_to_params():
    """BashTool.to_params returns the fixed Anthropic bash_20241022 schema."""
    tool = BashTool()
    assert tool.to_params() == {"type": "bash_20241022", "name": "bash"}


def test_bash_tool_call_runs_session_command():
    """BashTool with a command starts a session and delegates execution to that session."""
    tool = BashTool()
    session = mock.AsyncMock()
    session.run = mock.AsyncMock(return_value=ToolResult(output="done"))
    with mock.patch.object(_bash, "_BashSession", return_value=session):
        result = asyncio.run(tool(command="echo hi"))
    session.start.assert_awaited_once()
    session.run.assert_awaited_once_with("echo hi")
    assert result.output == "done"


def test_bash_tool_restart_stops_existing_session():
    """BashTool restart=True stops the current session before starting a fresh one."""
    tool = BashTool()
    old_session = mock.Mock()
    tool._session = old_session
    new_session = mock.AsyncMock()
    with mock.patch.object(_bash, "_BashSession", return_value=new_session):
        result = asyncio.run(tool(restart=True))
    old_session.stop.assert_called_once()
    new_session.start.assert_awaited_once()
    assert "restarted" in result.system


def test_bash_session_start_is_noop_when_started():
    """_BashSession.start returns immediately if the session is already started, keeping its process."""
    session = _bash._BashSession()
    session._started = True
    session._process = mock.Mock()
    asyncio.run(session.start())
    assert session._process is not None


def test_bash_session_stop_before_start_raises():
    """_BashSession.stop raises ToolError when the session was never started."""
    session = _bash._BashSession()
    with pytest.raises(ToolError, match="has not started"):
        session.stop()


def test_bash_session_stop_when_process_exited_is_noop():
    """_BashSession.stop does not terminate an already-exited process."""
    session = _bash._BashSession()
    session._started = True
    session._process = mock.Mock(returncode=0)
    session.stop()
    session._process.terminate.assert_not_called()


def test_bash_session_stop_terminates_running_process():
    """_BashSession.stop terminates a still-running process."""
    session = _bash._BashSession()
    session._started = True
    session._process = mock.Mock(returncode=None)
    session.stop()
    session._process.terminate.assert_called_once()


def test_bash_session_run_before_start_raises():
    """_BashSession.run raises ToolError when the session has not been started yet."""
    session = _bash._BashSession()
    with mock.patch.object(builtins, "input", return_value="yes"):
        with pytest.raises(ToolError, match="has not started"):
            asyncio.run(session.run("echo hi"))


def test_bash_session_run_cancelled_by_user():
    """_BashSession.run returns a cancellation ToolResult when the user does not approve the command."""
    session = _bash._BashSession()
    with mock.patch.object(builtins, "input", return_value="no"):
        result = asyncio.run(session.run("rm -rf /"))
    assert result.error == "User did not provide permission to execute the command."


def test_bash_session_run_when_process_exited_requires_restart():
    """_BashSession.run reports that bash must be restarted once the underlying process has exited."""
    session = _bash._BashSession()
    session._started = True
    session._process = mock.Mock(returncode=1)
    with mock.patch.object(builtins, "input", return_value="yes"):
        result = asyncio.run(session.run("echo hi"))
    assert result.error == "bash has exited with returncode 1"
    assert "restarted" in result.system


def test_bash_session_run_timeout_marks_session_stale():
    """_BashSession.run flags the session as timed out and raises ToolError when the sentinel never arrives."""
    session = _bash._BashSession()
    session._started = True
    session._process = _FakeProcess()
    with mock.patch.object(builtins, "input", return_value="yes"):
        with mock.patch.object(asyncio, "timeout", side_effect=asyncio.TimeoutError):
            with pytest.raises(ToolError, match="timed out"):
                asyncio.run(session.run("sleep 100"))
    assert session._timed_out is True


def test_bash_session_run_strips_sentinel_and_trailing_newline():
    """_BashSession.run returns the output up to the sentinel, without the trailing newline, and clears buffers."""
    session = _bash._BashSession()
    session._started = True
    session._process = _FakeProcess(stdout=b"hello world\n<<exit>>\n", stderr=b"")
    with mock.patch.object(builtins, "input", return_value="yes"):
        result = asyncio.run(session.run("echo hello"))
    assert result.output == "hello world"
    assert result.error == ""
    assert session._process.stdout._buffer == bytearray()
    assert session._process.stderr._buffer == bytearray()
    assert session._process.stdin.written.endswith(b"'<<exit>>'\n")
