"""Tests for semgrep-based scan_code without requiring semgrep on PATH."""

from types import SimpleNamespace
from unittest import mock

from interpreter.core.utils import scan_code


def test_scan_code_runs_and_cleans_up(tmp_path):
    """scan_code writes code to a temp file, runs semgrep, and always cleans up afterward."""
    interpreter = SimpleNamespace(
        verbose=False,
        safe_mode="auto",
        computer=SimpleNamespace(
            terminal=SimpleNamespace(
                get_language=lambda lang: SimpleNamespace(
                    file_extension="py", name="Python"
                )
            )
        ),
    )
    temp_path = str(tmp_path / "scan.py")

    with mock.patch("shutil.which", return_value="/usr/bin/semgrep"):
        with mock.patch(
            "interpreter.core.utils.scan_code.create_temporary_file",
            return_value=temp_path,
        ) as create_temp:
            with mock.patch(
                "interpreter.core.utils.scan_code.cleanup_temporary_file"
            ) as cleanup:
                with mock.patch(
                    "interpreter.core.utils.scan_code.subprocess.run",
                    return_value=mock.Mock(returncode=0),
                ) as run:
                    mock_spinner = mock.Mock()
                    mock_spinner.__enter__ = mock.Mock(return_value=mock_spinner)
                    mock_spinner.__exit__ = mock.Mock(return_value=False)
                    with mock.patch.object(scan_code, "yaspin", create=True) as yaspin:
                        yaspin.return_value.green.right.binary = mock_spinner
                        scan_code.scan_code("print(1)", "python", interpreter)

    create_temp.assert_called_once_with("print(1)", "py", verbose=False)
    run.assert_called_once()
    cmd = run.call_args[0][0]
    assert cmd[0] == "semgrep"
    assert cmd[-1] == "scan.py"
    cleanup.assert_called_once_with(temp_path, verbose=False)


def _scan(interpreter, monkeypatch, *, returncode=0, side_effect=None):
    """Run scan_code with temp files, subprocess and yaspin stubbed out.

    Returns the (create, run, cleanup) mocks so callers can assert the command
    and the always-run cleanup without touching semgrep or the real spinner.
    """
    temp_path = "/tmp/fake_scan.py"
    create = mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file",
        return_value=temp_path,
    )
    cleanup = mock.patch("interpreter.core.utils.scan_code.cleanup_temporary_file")
    run = mock.patch(
        "interpreter.core.utils.scan_code.subprocess.run",
        return_value=mock.Mock(returncode=returncode),
        side_effect=side_effect,
    )
    which = mock.patch("shutil.which", return_value="/usr/bin/semgrep")
    spinner = mock.Mock()
    spinner.__enter__ = mock.Mock(return_value=spinner)
    spinner.__exit__ = mock.Mock(return_value=False)
    yaspin = mock.patch.object(scan_code, "yaspin", create=True)
    with create as c, cleanup as cl, run as r, which, yaspin as y:
        y.return_value.green.right.binary = spinner
        scan_code.scan_code("print(1)", "python", interpreter)
    return c, r, cl


def _interpreter(safe_mode="auto", verbose=False):
    """A minimal interpreter stand-in for scan_code."""
    return SimpleNamespace(
        verbose=verbose,
        safe_mode=safe_mode,
        computer=SimpleNamespace(
            terminal=SimpleNamespace(
                get_language=lambda lang: SimpleNamespace(
                    file_extension="py", name="Python"
                )
            )
        ),
    )


def test_semgrep_command_targets_the_temp_file_in_its_directory():
    """The semgrep command runs without a shell, in the temp file's directory.

    The temp file is passed by basename with cwd= set to its directory, so a
    temp path containing spaces (e.g. a Windows profile temp dir) works. A
    shell-string command with `cd ... &&` or `shell=True` would break there
    (issue #375).
    """
    temp_path = "/tmp/dir with spaces/fake_scan.py"
    with mock.patch("shutil.which", return_value="/usr/bin/semgrep"), mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file", return_value=temp_path
    ), mock.patch("interpreter.core.utils.scan_code.cleanup_temporary_file"), \
        mock.patch(
            "interpreter.core.utils.scan_code.subprocess.run",
            return_value=mock.Mock(returncode=0),
        ) as run, mock.patch.object(scan_code, "yaspin", create=True) as yaspin:
        spinner = mock.Mock()
        spinner.__enter__ = mock.Mock(return_value=spinner)
        spinner.__exit__ = mock.Mock(return_value=False)
        yaspin.return_value.green.right.binary = spinner
        scan_code.scan_code("print(1)", "python", _interpreter())

    cmd = run.call_args[0][0]
    assert isinstance(cmd, list)
    assert cmd[:2] == ["semgrep", "scan"]
    assert cmd[-1] == "fake_scan.py"
    assert run.call_args.kwargs["cwd"] == "/tmp/dir with spaces"
    assert run.call_args.kwargs.get("shell") is not True


def test_clean_scan_reports_no_issues(capsys):
    """A zero exit code prints the no-issues line naming the language.

    Clients read this confirmation to know the scan completed cleanly; the
    returncode branch and the language name in the message are the contract.
    """
    _scan(_interpreter(safe_mode="auto"), None)

    out = capsys.readouterr().out
    assert "No issues were found in this Python code." in out
    assert "Code Scanner:" in out


def test_clean_scan_omits_scanner_prefix_outside_auto_mode(capsys):
    """Unless safe_mode is "auto", the message drops the "Code Scanner: " prefix.

    The prefix is conditional on safe_mode; an unconditional print would leak the
    scanner label into non-auto runs.
    """
    _scan(_interpreter(safe_mode="ask"), None)

    out = capsys.readouterr().out
    assert "No issues were found in this Python code." in out
    assert "Code Scanner:" not in out


def test_nonzero_returncode_prints_no_success_line(capsys):
    """A failing semgrep run does not print the no-issues message.

    The message is gated on returncode == 0; otherwise a clean report would be
    printed even when semgrep found problems.
    """
    _scan(_interpreter(), None, returncode=1)

    assert "No issues were found" not in capsys.readouterr().out


def test_scanner_exception_is_reported_and_cleaned_up(capsys):
    """An exception raised by semgrep is swallowed, reported, and cleanup still runs.

    Semgrep is optional, so a missing binary must not break the interpreter; the
    error line tells the user to install it and the temp file is still removed.
    """
    interpreter = _interpreter()
    temp_path = "/tmp/fake_scan.py"
    with mock.patch("shutil.which", return_value="/usr/bin/semgrep"), mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file", return_value=temp_path
    ), mock.patch("interpreter.core.utils.scan_code.cleanup_temporary_file") as cleanup, \
        mock.patch(
            "interpreter.core.utils.scan_code.subprocess.run",
            side_effect=FileNotFoundError("semgrep"),
        ), mock.patch.object(scan_code, "yaspin", create=True) as yaspin:
        spinner = mock.Mock()
        spinner.__enter__ = mock.Mock(return_value=spinner)
        spinner.__exit__ = mock.Mock(return_value=False)
        yaspin.return_value.green.right.binary = spinner
        scan_code.scan_code("print(1)", "python", interpreter)

    out = capsys.readouterr().out
    assert "Could not scan python code" in out
    assert "semgrep" in out
    cleanup.assert_called_once_with(temp_path, verbose=False)


def test_missing_semgrep_prints_guidance_without_running(capsys):
    """With no semgrep binary, the user is told to install it and nothing runs (issue #373).

    Previously the shell exited nonzero and subprocess.run returned normally,
    so nothing was printed at all — the user could not tell whether the scan
    ran, passed, or was skipped.
    """
    interpreter = _interpreter()
    with mock.patch("shutil.which", return_value=None), mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file"
    ) as create_temp, mock.patch(
        "interpreter.core.utils.scan_code.subprocess.run"
    ) as run:
        scan_code.scan_code("print(1)", "python", interpreter)

    out = capsys.readouterr().out
    assert "Could not scan python code" in out
    assert "semgrep" in out
    create_temp.assert_not_called()
    run.assert_not_called()


def test_temp_creation_failure_skips_scan(capsys):
    """A temp-file failure skips the optional scan instead of aborting the turn (issue #374).

    create_temporary_file raises on failure (PR #433); scan_code must not let
    that propagate out of an optional safe-mode nicety.
    """
    interpreter = _interpreter()
    with mock.patch("shutil.which", return_value="/usr/bin/semgrep"), mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file",
        side_effect=OSError("No space left on device"),
    ), mock.patch(
        "interpreter.core.utils.scan_code.cleanup_temporary_file"
    ) as cleanup, mock.patch(
        "interpreter.core.utils.scan_code.subprocess.run"
    ) as run:
        scan_code.scan_code("print(1)", "python", interpreter)  # must not raise

    run.assert_not_called()
    cleanup.assert_not_called()
    assert "No issues were found" not in capsys.readouterr().out
