"""Regression tests for #399: temp-file failures must propagate.

create_temporary_file swallowed every exception and returned None, so a
full disk surfaced later as an unrelated TypeError from
os.path.dirname(None) inside scan_code. The original error now
propagates after the diagnostic print.
"""

from unittest import mock

import pytest

from interpreter.core.utils.temporary_file import create_temporary_file


def test_creation_failure_reraises_original_error(capsys):
    """An OSError while writing still reaches the caller intact."""
    with mock.patch(
        "tempfile.NamedTemporaryFile", side_effect=OSError("disk full")
    ):
        with pytest.raises(OSError, match="disk full"):
            create_temporary_file("x = 1", "py")

    out = capsys.readouterr().out
    assert "Could not create temporary file." in out


def test_creation_success_returns_path():
    """The happy path still returns a real file holding the contents."""
    path = create_temporary_file("x = 1", "py")

    try:
        with open(path) as handle:
            assert handle.read() == "x = 1"
    finally:
        import os

        os.remove(path)
