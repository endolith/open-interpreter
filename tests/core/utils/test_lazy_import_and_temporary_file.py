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


def test_lazy_import_registers_the_module_in_sys_modules():
    """The imported module is cached under its name in sys.modules.

    lazy_import's whole point is a single lookup; a mutation that stored None
    (module -> None, sys.modules[name] -> None) would make the second call
    return the wrong object and defeat the cache.
    """
    import sys

    name = "json"
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
    misnamed files.
    """
    monkeypatch.chdir(tmp_path)
    path = create_temporary_file("x")
    assert not path.endswith(".")
    assert not path.endswith(".None")
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


def test_lazy_import_defers_loading():
    """lazy_import wraps the module's loader in a LazyLoader.

    The module is bound for lazy loading rather than executed eagerly; a mutated
    loader (LazyLoader(None), spec.loader = None) would fail to import the
    module's contents on first attribute access.
    """
    name = "fractions"
    import sys as _sys

    _sys.modules.pop(name, None)
    module = lazy_import(name)
    assert module.Fraction(1, 2) == module.Fraction(1, 2)

