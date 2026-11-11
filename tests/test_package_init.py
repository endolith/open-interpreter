"""Import-time tests for the ``interpreter`` package init.

The ``--os`` branch (update check, voice flag, async entrypoint) only
executes during a fresh package import with ``--os`` in ``sys.argv``, so
these tests evict the package from ``sys.modules`` and re-import it.
"""

import importlib
import subprocess
import sys
import textwrap
from types import ModuleType
from unittest import mock

import pytest
import requests


def _evict_interpreter_package():
    """Remove the interpreter package tree; return it for restoration."""
    evicted = {
        name: module
        for name, module in sys.modules.items()
        if name == "interpreter" or name.startswith("interpreter.")
    }
    for name in evicted:
        del sys.modules[name]
    return evicted


def _stub_entrypoint(monkeypatch):
    """Stub the computer-use async entrypoint so --os import stays headless."""
    package = ModuleType("interpreter.computer_use")
    package.__path__ = []
    loop = ModuleType("interpreter.computer_use.loop")
    loop.run_async_main = mock.Mock()
    monkeypatch.setitem(sys.modules, "interpreter.computer_use", package)
    monkeypatch.setitem(sys.modules, "interpreter.computer_use.loop", loop)
    return loop


def _run_python(code):
    """Run a snippet in a fresh interpreter and return the finished process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def test_plain_import_defers_singleton_and_heavy_imports():
    """`import interpreter` must not build the interpreter or import litellm.

    The CLI imports this package on every invocation, so constructing the
    singleton here would put litellm and the whole toolbox on the startup path
    of `--help` and `--version`. A fresh subprocess is used because the test
    session itself has long since imported everything.
    """
    proc = _run_python(
        """
        import sys
        import interpreter

        assert "interpreter.core.core" not in sys.modules
        assert "litellm" not in sys.modules
        assert interpreter._singleton is None
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_singleton_access_builds_one_cached_interpreter():
    """PEP 562 access to `interpreter` builds an OpenInterpreter once and reuses it."""
    proc = _run_python(
        """
        from interpreter import OpenInterpreter
        from interpreter import interpreter as first
        from interpreter import interpreter as second

        assert isinstance(first, OpenInterpreter)
        assert first is second
        assert first is __import__("interpreter")._singleton
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_toolbox_and_ai2_are_singleton_owned():
    """`toolbox` and `ai2` resolve to the objects the singleton already holds."""
    proc = _run_python(
        """
        from interpreter import ai2, interpreter, toolbox

        assert toolbox is interpreter.toolbox
        assert ai2 is interpreter.toolbox.ai2
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_public_classes_resolve_through_the_module():
    """The public classes remain importable from the package after the split."""
    proc = _run_python(
        """
        from interpreter import AsyncInterpreter, BaseLanguage, OpenInterpreter

        assert issubclass(AsyncInterpreter, OpenInterpreter)
        assert BaseLanguage.__name__ == "BaseLanguage"
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_unknown_attribute_raises_attribute_error():
    """An unknown package attribute fails loudly instead of returning a stale name."""
    proc = _run_python(
        """
        import interpreter

        try:
            interpreter.definitely_not_a_public_name
        except AttributeError as error:
            assert "has no attribute" in str(error)
        else:
            raise AssertionError("expected AttributeError")
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


@pytest.mark.xfail(
    raises=TypeError,
    reason="interpreter/__init__.py rebinds `version` via "
    "`from packaging import version`, so check_for_update() calls the "
    "module and every --os import dies with TypeError (#407; fix in #430)",
    strict=False,
)
def test_os_flag_checks_for_update_then_runs_main(monkeypatch, capsys):
    """--os import prints an update notice for newer PyPI releases, then
    runs the async entrypoint and exits instead of building an interpreter."""
    evicted = _evict_interpreter_package()
    try:
        loop = _stub_entrypoint(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["oi", "--os"])
        response = mock.Mock()
        response.json.return_value = {"info": {"version": "99.0.0"}}
        monkeypatch.setattr(
            requests, "get", lambda *a, **k: response
        )

        with pytest.raises(SystemExit):
            importlib.import_module("interpreter")

        loop.run_async_main.assert_called_once_with()
        assert "new version" in capsys.readouterr().out
    finally:
        for name in [
            name
            for name in sys.modules
            if name == "interpreter" or name.startswith("interpreter.")
        ]:
            del sys.modules[name]
        sys.modules.update(evicted)
