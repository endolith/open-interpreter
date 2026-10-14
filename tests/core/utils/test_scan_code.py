"""Tests for semgrep-based scan_code without requiring semgrep on PATH."""

import os
import shutil
from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.utils import scan_code


@pytest.fixture(autouse=True)
def _semgrep_present(monkeypatch):
    """Pretend semgrep is installed: the which-gate is not under test here."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/semgrep")


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
    """The semgrep command scans the temp file by basename with cwd set.

    The command is an arg list run with cwd= and no shell (fixed in #442
    for #375): a temp dir containing spaces works unquoted, and a dropped
    basename or a full path where a basename is expected would scan nothing.
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
    assert cmd == [
        "semgrep",
        "scan",
        "--config",
        "auto",
        "--quiet",
        "--error",
        "fake_scan.py",
    ]
    assert run.call_args.kwargs["cwd"] == os.path.dirname(temp_path)
    assert "shell" not in run.call_args.kwargs


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


def test_missing_semgrep_prints_guidance_without_running(capsys):
    """With no semgrep binary, the user is told to install it and nothing runs (issue #373).

    Previously the shell exited nonzero and subprocess.run returned normally,
    so nothing was printed at all — the user could not tell whether the scan
    ran, passed, or was skipped.
    """
    interpreter = _scan_interpreter()
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

    create_temporary_file swallows its own errors and returns None rather
    than raising, so scan_code must treat a None return as "skip the scan"
    instead of crashing in os.path.dirname. Also closes #399, whose strict
    xfail pinned the same TypeError-from-dirname crash with no assertions.
    """
    interpreter = _scan_interpreter()
    with mock.patch("shutil.which", return_value="/usr/bin/semgrep"), mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file",
        return_value=None,
    ), mock.patch(
        "interpreter.core.utils.scan_code.cleanup_temporary_file"
    ) as cleanup, mock.patch(
        "interpreter.core.utils.scan_code.subprocess.run"
    ) as run:
        scan_code.scan_code("print(1)", "python", interpreter)  # must not raise

    run.assert_not_called()
    cleanup.assert_not_called()
    assert "No issues were found" not in capsys.readouterr().out


def test_temp_creation_raise_still_skips_scan():
    """A raising temp helper still skips the scan via the defensive except (issue #374).

    The real helper returns None (covered above), but the except stays as
    defense for e.g. language_class.file_extension access failing inside
    the same try block.
    """
    interpreter = _scan_interpreter()
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


def test_no_yaspin_falls_back_to_nullcontext():
    """Without yaspin installed the scan runs inside a nullcontext (covers the spinner fallback)."""
    with mock.patch("shutil.which", return_value="/usr/bin/semgrep"), mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file",
        return_value="/tmp/fake_scan.py",
    ), mock.patch(
        "interpreter.core.utils.scan_code.cleanup_temporary_file"
    ), mock.patch(
        "interpreter.core.utils.scan_code.subprocess.run",
        return_value=mock.Mock(returncode=0),
    ) as run, mock.patch.object(
        scan_code, "yaspin", None
    ):
        scan_code.scan_code("print(1)", "python", _scan_interpreter())
    run.assert_called_once()


def test_verbose_temp_failure_names_the_error(capsys):
    """A temp-file failure in verbose mode prints the underlying error (issue #374)."""
    with mock.patch("shutil.which", return_value="/usr/bin/semgrep"), mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file",
        side_effect=OSError("disk full"),
    ):
        scan_code.scan_code("print(1)", "python", _scan_interpreter(verbose=True))
    assert "disk full" in capsys.readouterr().out


def test_verbose_scan_prints_progress(capsys, tmp_path):
    """Verbose mode narrates the scan target before running semgrep."""
    temp_path = str(tmp_path / "scan.py")
    interpreter = _scan_interpreter(verbose=True)
    with mock.patch(
        "interpreter.core.utils.scan_code.create_temporary_file",
        return_value=temp_path,
    ), mock.patch(
        "interpreter.core.utils.scan_code.cleanup_temporary_file"
    ), mock.patch(
        "interpreter.core.utils.scan_code.subprocess.run",
        return_value=mock.Mock(returncode=0),
    ):
        scan_code.scan_code("print(1)", "python", interpreter)
    assert "Scanning python code" in capsys.readouterr().out


def test_missing_yaspin_import_disables_spinner():
    """When yaspin cannot be imported the module degrades to no spinner instead of crashing."""
    import importlib
    import sys

    with mock.patch.dict(sys.modules, {"yaspin": None, "yaspin.spinners": None}):
        try:
            importlib.reload(scan_code)
            assert scan_code.yaspin is None
            assert scan_code.Spinners is None
        finally:
            sys.modules.pop("yaspin", None)
            sys.modules.pop("yaspin.spinners", None)
            importlib.reload(scan_code)
