"""Characterization tests for ``computer.docs``.

``docs.py`` uses the optional ``aifs`` integration. Each test patches the
module's ``aifs`` reference so it can verify ``Docs.search`` behavior without
requiring that dependency.
"""

import inspect
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.computer.docs import docs as docs_mod
from interpreter.core.computer.docs.docs import Docs


def test_search_delegates_to_aifs_with_file_paths():
    """Docs.search(paths=...) forwards the explicit file paths to aifs."""
    docs = Docs(SimpleNamespace())
    paths = ["/a.py", "/b.py"]
    with mock.patch.object(docs_mod, "aifs", mock.Mock()) as aifs:
        result = docs.search("query", paths=paths)

    aifs.search.assert_called_once_with(
        "query", file_paths=paths, python_docstrings_only=True
    )
    assert result == aifs.search.return_value


def test_search_scans_module_source_directory():
    """Docs.search() defaults to scanning the directory of the given module's
    source file (the computer's class file when no module is given)."""
    docs = Docs(SimpleNamespace())
    expected = os.path.dirname(inspect.getfile(SimpleNamespace))
    with mock.patch.object(docs_mod, "aifs", mock.Mock()) as aifs:
        docs.search("query")

    aifs.search.assert_called_once_with(
        "query", path=expected, python_docstrings_only=True
    )


def test_search_with_explicit_module_uses_its_directory():
    """Docs.search(module=...) scans the directory of that module's class file."""

    class DummyModule:
        pass

    docs = Docs(SimpleNamespace())
    expected = os.path.dirname(inspect.getfile(DummyModule))
    with mock.patch.object(docs_mod, "aifs", mock.Mock()) as aifs:
        docs.search("query", module=DummyModule())

    aifs.search.assert_called_once_with(
        "query", path=expected, python_docstrings_only=True
    )


def test_search_requires_aifs_installed():
    """Docs.search() without aifs installed fails loudly rather than silently
    returning nothing."""
    docs = Docs(SimpleNamespace())
    with mock.patch.object(docs_mod, "aifs", None):
        with pytest.raises(AttributeError):
            docs.search("query")
