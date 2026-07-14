from unittest import mock

import interpreter.core.computer.utils.run_applescript as run_applescript


def test_run_applescript_returns_stdout(monkeypatch):
    """run_applescript returns the stdout captured from osascript."""
    monkeypatch.setattr(
        run_applescript.subprocess,
        "check_output",
        lambda args, **kwargs: "hello\n",
    )

    assert run_applescript.run_applescript('display dialog "hi"') == "hello\n"


def test_run_applescript_capture_returns_stdout_and_stderr(monkeypatch):
    """run_applescript_capture returns the captured (stdout, stderr) pair."""

    class _Completed:
        stdout = "out"
        stderr = "err"

    monkeypatch.setattr(
        run_applescript.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(),
    )

    stdout, stderr = run_applescript.run_applescript_capture('get name')
    assert stdout == "out"
    assert stderr == "err"
