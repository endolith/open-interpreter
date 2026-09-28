"""Tests for semgrep-based scan_code without requiring semgrep on PATH."""

import contextlib
import os
from types import SimpleNamespace
from unittest import mock

from interpreter.core.utils import scan_code


def _interpreter():
    """The slice of the interpreter that scan_code reads."""
    return SimpleNamespace(
        verbose=False,
        safe_mode="auto",
        computer=SimpleNamespace(
            terminal=SimpleNamespace(get_language=lambda lang: SimpleNamespace(file_extension="py", name="Python"))
        ),
    )


@contextlib.contextmanager
def _scan_patches(run):
    """Silence the spinner and stand in for the semgrep subprocess.

    `run` replaces subprocess.run, so a test can choose the scan's exit status
    or make it raise, without semgrep or a terminal being present.
    """
    spinner = mock.Mock()
    spinner.__enter__ = mock.Mock(return_value=spinner)
    spinner.__exit__ = mock.Mock(return_value=False)
    with (
        mock.patch.object(scan_code, "yaspin", create=True) as yaspin,
        mock.patch("interpreter.core.utils.scan_code.subprocess.run", side_effect=run) as run_mock,
    ):
        yaspin.return_value.green.right.binary = spinner
        yield run_mock


def test_scan_code_runs_and_cleans_up(tmp_path):
    """scan_code writes code to a temp file, runs semgrep, and always cleans up afterward."""
    interpreter = _interpreter()
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


def test_clean_scan_reports_no_issues_and_names_the_language(capsys):
    """A zero exit status prints the reassurance line, naming the scanned language.

    In "auto" mode nothing else confirms the scan ran, so without this line the
    user cannot tell scanning from not scanning.
    """
    with _scan_patches(lambda *args, **kwargs: SimpleNamespace(returncode=0)):
        scan_code.scan_code("print(1)", "python", _interpreter())

    output = capsys.readouterr().out
    assert "No issues were found in this Python code" in output
    assert "Code Scanner:" in output


def test_a_failing_scan_does_not_claim_the_code_is_clean(capsys):
    """A non-zero exit status stays quiet rather than contradicting semgrep.

    semgrep prints its own findings to the terminal, so an added "no issues"
    line would dispute what the user just read.
    """
    with _scan_patches(lambda *args, **kwargs: SimpleNamespace(returncode=1)):
        scan_code.scan_code("print(1)", "python", _interpreter())

    assert "No issues were found" not in capsys.readouterr().out


def test_a_scan_launch_failure_is_reported_and_does_not_stop_the_turn(capsys):
    """An exception while launching the scan is reported rather than raised.

    Safe mode is opt-in and semgrep is an optional dependency, so a failure
    here must not take the turn down; the handler prints the semgrep hint and
    the scan is skipped. A semgrep binary that is missing altogether does not
    reach this handler — with shell=True the shell exits nonzero instead, which
    is the quiet path covered above.
    """

    def _launch_failure(*args, **kwargs):
        raise OSError("failed to launch shell")

    with _scan_patches(_launch_failure):
        scan_code.scan_code("print(1)", "python", _interpreter())

    assert "Have you installed 'semgrep'?" in capsys.readouterr().out


def test_the_scanned_file_does_not_outlive_the_scan():
    """The scratch copy of the code is gone once the scan finishes.

    It holds whatever the model was about to run — possibly including secrets —
    and the temp directory is world-readable on most systems, so the file
    semgrep was pointed at must not be left behind.
    """
    real_create = scan_code.create_temporary_file
    created = {}

    def _recording_create(contents, extension=None, verbose=False):
        path = real_create(contents, extension, verbose=verbose)
        created["path"] = path
        return path

    with (
        mock.patch.object(scan_code, "create_temporary_file", side_effect=_recording_create),
        _scan_patches(lambda *args, **kwargs: SimpleNamespace(returncode=0)),
    ):
        scan_code.scan_code("secret = 'value'", "python", _interpreter())

    assert created["path"] is not None
    assert not os.path.exists(created["path"])
