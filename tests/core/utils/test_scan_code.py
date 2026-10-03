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
    assert "semgrep" in cmd
    assert "scan.py" in cmd
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
    spinner = mock.Mock()
    spinner.__enter__ = mock.Mock(return_value=spinner)
    spinner.__exit__ = mock.Mock(return_value=False)
    yaspin = mock.patch.object(scan_code, "yaspin", create=True)
    with create as c, cleanup as cl, run as r, yaspin as y:
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
    """The semgrep command cds into the temp dir and scans the temp file by name.

    The command must run from the file's directory so semgrep resolves its config
    relative to the file; a dropped `cd`, wrong flags, or a full path where a
    basename is expected would scan nothing.
    """
    temp_path = "/tmp/fake_scan.py"
    with mock.patch(
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
    assert cmd.startswith("cd /tmp &&")
    assert "semgrep scan --config auto --quiet --error fake_scan.py" in cmd
    assert run.call_args.kwargs["shell"] is True


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
    with mock.patch(
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
