"""Import-time tests for the ``interpreter`` package init.

The ``--os`` branch (update check, voice flag, async entrypoint) only
executes during a fresh package import with ``--os`` in ``sys.argv``, so
these tests evict the package from ``sys.modules`` and re-import it.
"""

import importlib
import sys
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


def test_os_flag_checks_for_update_then_runs_main(monkeypatch, capsys):
    """--os import prints an update notice for newer PyPI releases, then
    runs the async entrypoint and exits instead of building an interpreter.

    Regression test for #407: `check_for_update()` used to call the packaging
    module and die with TypeError before reaching any of this (fixed in #430
    by aliasing the import).
    """
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


def test_update_check_request_sets_a_timeout(monkeypatch):
    """The update check must not block --os startup on a stalled PyPI.

    A connection that opens but never responds would hang the import, so the
    request needs a timeout; the existing exception handler can only continue
    startup once the call returns.
    """
    evicted = _evict_interpreter_package()
    try:
        _stub_entrypoint(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["oi", "--os"])
        calls = {}

        def fake_get(*args, **kwargs):
            calls.update(kwargs)
            response = mock.Mock()
            response.json.return_value = {"info": {"version": "0.0.0"}}
            return response

        monkeypatch.setattr(requests, "get", fake_get)

        with pytest.raises(SystemExit):
            importlib.import_module("interpreter")

        assert calls.get("timeout") is not None
    finally:
        for name in [
            name
            for name in sys.modules
            if name == "interpreter" or name.startswith("interpreter.")
        ]:
            del sys.modules[name]
        sys.modules.update(evicted)
