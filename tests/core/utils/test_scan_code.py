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
