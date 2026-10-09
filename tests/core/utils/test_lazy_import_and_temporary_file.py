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


def test_lazy_import_default_is_optional():
    """Omitting optional defaults to True, returning None for a missing module.

    The default must stay True: a False default would turn every optional import
    into a hard failure, breaking startup when an optional dependency is absent.
    """
    assert lazy_import("this_module_definitely_does_not_exist_xyz") is None


def test_lazy_import_registers_the_module_in_sys_modules(monkeypatch):
    """The imported module is cached under its name in sys.modules.

    lazy_import's whole point is a single lookup; a mutation that stored None
    (module -> None, sys.modules[name] -> None) would make the second call
    return the wrong object and defeat the cache.

    Uses a module that is not already in sys.modules, so the call has to take
    the registration branch instead of returning an existing entry. "json" is
    imported by pytest itself, which left that branch unreached here.
    """
    import sys

    name = "fractions"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = lazy_import(name)

    assert sys.modules[name] is module
    assert lazy_import(name) is module


def test_create_temporary_file_verbose_defaults_to_silent(capsys, tmp_path, monkeypatch):
    """Omitting verbose leaves create and cleanup silent.

    A True default would print to stdout from every temp-file use, so the
    default is pinned by asserting nothing is printed.
    """
    monkeypatch.chdir(tmp_path)
    path = create_temporary_file("quiet", extension="txt")
    assert capsys.readouterr().out == ""
    cleanup_temporary_file(path)
    assert capsys.readouterr().out == ""


def test_extension_defaults_to_no_suffix(tmp_path, monkeypatch):
    """With no extension the file has no ".None" or trailing-dot suffix.

    The suffix is f".{extension}" when extension is truthy and "" otherwise; a
    mutation that appended ".None" or dropped the conditional would create
    misnamed files. Asserting the extracted suffix is empty rather than the
    absence of two specific endings, so an unrelated appended suffix is caught
    too.
    """
    monkeypatch.chdir(tmp_path)
    path = create_temporary_file("x")
    assert os.path.splitext(path)[1] == ""
    cleanup_temporary_file(path)


def test_extension_is_appended_as_a_dot_suffix(tmp_path, monkeypatch):
    """A provided extension becomes a ".py"-style suffix on the file name.

    The suffix must use the extension verbatim; a wrapped or dropped extension
    would produce files the language tooling cannot recognise.
    """
    monkeypatch.chdir(tmp_path)
    path = create_temporary_file("print(1)", extension="py")
    assert path.endswith(".py")
    cleanup_temporary_file(path)


def test_lazy_import_defers_loading(tmp_path, monkeypatch):
    """lazy_import does not execute the module body until an attribute is read.

    The earlier version of this test only checked the module's contents work,
    which passes identically for eager and deferred loading and so pinned
    nothing about deferral. Here the module body has an observable side effect
    (it writes a marker file), which lets the test assert the body has *not*
    run after lazy_import and *has* run once an attribute is touched.
    """
    import sys

    marker = tmp_path / "executed.marker"
    module_dir = tmp_path / "probe"
    module_dir.mkdir()
    (module_dir / "oi_lazy_probe.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text('executed')\n"
        "VALUE = 42\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(module_dir))

    name = "oi_lazy_probe"
    monkeypatch.delitem(sys.modules, name, raising=False)

    module = lazy_import(name)
    assert not marker.exists(), "module body ran during lazy_import; loading was eager"

    assert module.VALUE == 42
    assert marker.exists(), "reading an attribute did not execute the module body"

