import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

_tools = Path(__file__).resolve().parents[3] / "interpreter/computer_use/tools"

spec = importlib.util.spec_from_file_location(
    "interpreter.computer_use.tools.run", _tools / "run.py"
)
_run = importlib.util.module_from_spec(spec)
sys.modules["interpreter.computer_use.tools.run"] = _run
spec.loader.exec_module(_run)

TRUNCATED_MESSAGE = _run.TRUNCATED_MESSAGE
maybe_truncate = _run.maybe_truncate
run = _run.run


def test_maybe_truncate_leaves_short_content():
    assert maybe_truncate("hello", truncate_after=10) == "hello"


def test_maybe_truncate_clips_long_content():
    content = "x" * 20
    result = maybe_truncate(content, truncate_after=10)
    assert result.startswith("x" * 10)
    assert result.endswith(TRUNCATED_MESSAGE)


def test_run_returns_stdout_stderr():
    async def fake_communicate():
        return (b"out", b"err")

    mock_process = mock.Mock()
    mock_process.communicate = fake_communicate
    mock_process.returncode = 0

    with mock.patch.object(
        _run.asyncio, "create_subprocess_shell", return_value=mock_process
    ):
        code, stdout, stderr = asyncio.run(run("echo hi"))
    assert code == 0
    assert stdout == "out"
    assert stderr == "err"


def test_run_raises_timeout_error():
    # communicate is a plain Mock (not async) because wait_for is patched to
    # raise TimeoutError before it ever tries to await the communicate()
    # result. Using a plain Mock avoids an unawaited-coroutine RuntimeWarning
    # that AsyncMock would trigger when wait_for discards it.
    mock_process = mock.Mock()
    mock_process.communicate = mock.Mock(return_value=None)
    mock_process.kill = mock.Mock()

    with mock.patch.object(
        _run.asyncio, "create_subprocess_shell", return_value=mock_process
    ):
        with mock.patch.object(
            _run.asyncio, "wait_for", side_effect=asyncio.TimeoutError
        ):
            with pytest.raises(TimeoutError, match="timed out"):
                asyncio.run(run("sleep 999", timeout=0.1))
