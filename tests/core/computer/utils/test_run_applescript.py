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


def test_run_applescript_uses_osascript_with_e_flag_and_text_mode(monkeypatch):
    """run_applescript invokes `osascript -e <script>` with universal_newlines.

    The binary name, the -e flag, and text mode are the whole call contract; a
    renamed binary or flag would fail on macOS, and dropping universal_newlines
    would return bytes instead of str.
    """
    seen = {}

    def fake_check_output(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(run_applescript.subprocess, "check_output", fake_check_output)

    run_applescript.run_applescript('display dialog "hi"')

    assert seen["args"] == ["osascript", "-e", 'display dialog "hi"']
    assert seen["kwargs"] == {"universal_newlines": True}


def test_run_applescript_capture_uses_osascript_and_captures_text(monkeypatch):
    """run_applescript_capture runs `osascript -e <script>` capturing text output.

    capture_output/text keep stdout and stderr separable and decoded, and
    check=False lets the caller read stderr instead of raising on AppleScript
    errors; each keyword is part of that contract.
    """
    seen = {}

    class _Completed:
        stdout = "out"
        stderr = "err"

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return _Completed()

    monkeypatch.setattr(run_applescript.subprocess, "run", fake_run)

    run_applescript.run_applescript_capture('get name')

    assert seen["args"] == ["osascript", "-e", 'get name']
    assert seen["kwargs"] == {
        "capture_output": True,
        "text": True,
        "check": False,
    }

