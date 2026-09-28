import os
from unittest import mock

import pytest

from interpreter.core.utils.lazy_import import lazy_import
from interpreter.core.utils.temporary_file import (
    cleanup_temporary_file,
    create_temporary_file,
)


def test_create_temporary_file_writes_contents(tmp_path, monkeypatch):
    """create_temporary_file writes the given contents and cleanup_temporary_file removes the file."""
    monkeypatch.chdir(tmp_path)
    path = create_temporary_file("hello", extension="txt")
    assert os.path.exists(path)
    with open(path) as f:
        assert f.read() == "hello"
    cleanup_temporary_file(path)
    assert not os.path.exists(path)


def test_create_temporary_file_verbose(capsys, tmp_path, monkeypatch):
    """verbose=True prints creation and cleanup messages for temporary files."""
    monkeypatch.chdir(tmp_path)
    path = create_temporary_file("data", verbose=True)
    captured = capsys.readouterr()
    assert "Created temporary file" in captured.out
    cleanup_temporary_file(path, verbose=True)
    captured = capsys.readouterr()
    assert "Cleaning up temporary file" in captured.out


def test_cleanup_missing_file_does_not_raise(capsys):
    """cleanup_temporary_file on a missing path logs a warning instead of raising."""
    cleanup_temporary_file("/nonexistent/path/file.txt")
    captured = capsys.readouterr()
    assert "Could not clean up temporary file" in captured.out


def test_create_temporary_file_without_an_extension():
    """extension=None still yields a usable scratch file, with no suffix appended.

    Languages with no file extension rely on the scratch file being created
    without a stray separator in its name.
    """
    path = create_temporary_file("echo hi")
    try:
        assert os.path.isfile(path)
        assert os.path.splitext(path)[1] == ""
        with open(path) as f:
            assert f.read() == "echo hi"
    finally:
        cleanup_temporary_file(path)


def test_create_temporary_file_reports_failure_without_raising(capsys, monkeypatch):
    """A failure to create the scratch file returns None and explains itself.

    The caller is an optional safety feature (scan_code), so raising here would
    turn a missing temp directory into a crashed turn.
    """

    def _boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr("interpreter.core.utils.temporary_file.tempfile.NamedTemporaryFile", _boom)
    assert create_temporary_file("x", "py") is None
    assert "Could not create temporary file" in capsys.readouterr().out


def test_lazy_import_returns_existing_module():
    """lazy_import returns the already-imported module object for a valid module name."""
    import json as json_module

    assert lazy_import("json") is json_module


def test_lazy_import_optional_missing_returns_none():
    """lazy_import with optional=True returns None when the module cannot be imported."""
    assert lazy_import("this_module_definitely_does_not_exist_xyz", optional=True) is None


def test_lazy_import_required_missing_raises():
    """lazy_import with optional=False raises ImportError when the module cannot be found."""
    with pytest.raises(ImportError, match="cannot be found"):
        lazy_import("this_module_definitely_does_not_exist_xyz",
                    optional=False)
