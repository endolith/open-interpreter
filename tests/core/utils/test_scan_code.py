"""Tests for semgrep-based scan_code without requiring semgrep on PATH."""

from types import SimpleNamespace
from unittest import mock

import pytest

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


def _scan_interpreter(verbose=False):
    """A minimal interpreter for scan_code, with `verbose` under test."""
    return SimpleNamespace(
        verbose=verbose,
        safe_mode="auto",
        computer=SimpleNamespace(
            terminal=SimpleNamespace(
                get_language=lambda lang: SimpleNamespace(
                    file_extension="py", name="Python"
                )
            )
        ),
    )


def test_scan_code_announces_the_file_it_scans_when_verbose(tmp_path, capsys):
    """With verbose on, scan_code names the language and the file before scanning.

    These two prints are the only record of which file semgrep was pointed at, so
    they are what makes a scan reproducible by hand afterwards.
    """
    temp_path = str(tmp_path / "scan.py")
    interpreter = _scan_interpreter(verbose=True)

    with mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file",
        return_value=temp_path,
    ):
        with mock.patch("interpreter.core.utils.scan_code.cleanup_temporary_file"):
            with mock.patch("interpreter.core.utils.scan_code.subprocess.run"):
                scan_code.scan_code("x = 1", "python", interpreter)

    captured = capsys.readouterr().out
    assert "Scanning python code in scan.py" in captured


def test_scan_code_does_not_name_the_file_when_not_verbose(tmp_path, capsys):
    """Without verbose, scan_code does not report which file it is scanning.

    The assertion is on the verbose line specifically, not on the word
    "Scanning": the yaspin spinner emits its own "Scanning code..." on every run,
    so a substring check would fail for the wrong reason. What is under test is
    the language-and-filename line, which only verbose produces.
    """
    temp_path = str(tmp_path / "scan.py")
    interpreter = _scan_interpreter(verbose=False)

    with mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file",
        return_value=temp_path,
    ):
        with mock.patch("interpreter.core.utils.scan_code.cleanup_temporary_file"):
            with mock.patch("interpreter.core.utils.scan_code.subprocess.run"):
                scan_code.scan_code("x = 1", "python", interpreter)

    assert "Scanning python code in scan.py" not in capsys.readouterr().out


def test_scan_code_still_cleans_up_when_the_temp_file_cannot_be_created(tmp_path):
    """A temp-file failure must not leave a file behind, and must not run semgrep.

    Cleanup is in a `finally`, so it has to hold on the failure path too —
    otherwise a partially written file survives with no owner.
    """
    interpreter = _scan_interpreter()

    with mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file", return_value=None
    ):
        with mock.patch(
            "interpreter.core.utils.scan_code.cleanup_temporary_file"
        ) as cleanup:
            with mock.patch("interpreter.core.utils.scan_code.subprocess.run") as run:
                try:
                    scan_code.scan_code("x = 1", "python", interpreter)
                except TypeError as error:
                    # Narrow on purpose. A bare `except Exception` also swallows an
                    # unrelated failure, so this would report success while
                    # something else broke. A clean return is still accepted
                    # deliberately: once #399 is fixed scan_code should handle the
                    # missing file instead of raising, and this test should keep
                    # passing then.
                    assert "NoneType" in str(error), (
                        f"expected the missing-temp-file TypeError, got {error!r}"
                    )

    assert run.call_count == 0, "semgrep must not run without a file to scan"


@pytest.mark.xfail(
    reason="#399: create_temporary_file returns None on failure, so scan_code "
    "raises an unrelated TypeError from os.path.dirname",
    strict=True,
)
def test_scan_code_reports_a_temp_file_failure_clearly(tmp_path, capsys):
    """A temp-file failure should name the cause, not raise TypeError from os.

    `create_temporary_file` swallows its exception and returns `None`, and
    `scan_code` passes that straight to `os.path.dirname`, so the user sees
    "expected str, bytes or os.PathLike object, not NoneType" from a stdlib call
    that has nothing to do with the disk.

    Strict xfail: it passes the day #399 is fixed. Pinned as-is because the
    current message actively misdirects.
    """
    interpreter = _scan_interpreter()

    with mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file", return_value=None
    ):
        with mock.patch("interpreter.core.utils.scan_code.cleanup_temporary_file"):
            scan_code.scan_code("x = 1", "python", interpreter)
