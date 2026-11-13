"""Import-time tests for the ``interpreter`` package init.

The ``--os`` branch (update check, voice flag, async entrypoint) only
executes during a fresh package import with ``--os`` in ``sys.argv``, so
these tests evict the package from ``sys.modules`` and re-import it.
"""

import contextlib
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


@contextlib.contextmanager
def _fresh_os_import(monkeypatch):
    """Evict the package for a ``--os`` re-import and restore it afterwards.

    Yields the stubbed ``run_async_main``. The ``--os`` branch ends in
    ``exit()``, which propagates out of the import and drops the fresh module
    from ``sys.modules``; a test that needs the module object binds a side
    effect to ``run_async_main``, where it is still registered.
    """
    evicted = _evict_interpreter_package()
    try:
        # Undo the stubs before restoring the evicted modules below. The
        # ``monkeypatch`` fixture outlives this context manager, so its undo
        # would otherwise delete the restored ``interpreter.computer_use``
        # entries and leave later imports with a different module object than
        # the one collected tests hold.
        with monkeypatch.context() as stub_patches:
            loop = _stub_entrypoint(stub_patches)
            stub_patches.setattr(sys, "argv", ["oi", "--os"])
            yield loop
    finally:
        for name in [
            name
            for name in sys.modules
            if name == "interpreter" or name.startswith("interpreter.")
        ]:
            del sys.modules[name]
        sys.modules.update(evicted)


def test_fresh_os_import_keeps_an_evicted_module_after_the_outer_undo():
    """A module evicted before the re-import is still registered once the outer
    patch is undone.

    Regression test for the teardown order: the stub patches must be undone
    before the evicted modules are restored. Otherwise the outer
    ``monkeypatch`` undo, which outlives this context manager, deletes the
    restored ``interpreter.computer_use`` entries and the next import builds a
    second module object for the same name.
    """
    dummy = ModuleType("interpreter.computer_use")
    dummy.__path__ = []
    sys.modules["interpreter.computer_use"] = dummy
    try:
        with pytest.MonkeyPatch.context() as outer:
            with _fresh_os_import(outer):
                pass
        assert sys.modules.get("interpreter.computer_use") is dummy
    finally:
        sys.modules.pop("interpreter.computer_use", None)


def test_os_flag_checks_for_update_then_runs_main(monkeypatch, capsys):
    """--os import prints an update notice for newer PyPI releases, then
    runs the async entrypoint and exits instead of building an interpreter.

    Regression test for #407: `check_for_update()` used to call the packaging
    module and die with TypeError before reaching any of this (fixed in #430
    by aliasing the import).
    """
    response = mock.Mock()
    response.json.return_value = {"info": {"version": "99.0.0"}}

    with _fresh_os_import(monkeypatch) as loop:
        monkeypatch.setattr(requests, "get", lambda *a, **k: response)

        with pytest.raises(SystemExit):
            importlib.import_module("interpreter")

    loop.run_async_main.assert_called_once_with()
    assert "new version" in capsys.readouterr().out


def test_update_check_request_sets_a_timeout(monkeypatch):
    """The update check must not block --os startup on a stalled PyPI.

    A connection that opens but never responds would hang the import, so the
    request needs a timeout; the existing exception handler can only continue
    startup once the call returns.
    """
    calls = {}

    def fake_get(*args, **kwargs):
        calls.update(kwargs)
        response = mock.Mock()
        response.json.return_value = {"info": {"version": "0.0.0"}}
        return response

    with _fresh_os_import(monkeypatch):
        monkeypatch.setattr(requests, "get", fake_get)

        with pytest.raises(SystemExit):
            importlib.import_module("interpreter")

    assert calls.get("timeout") is not None


def test_update_check_failure_starts_os_mode_without_a_notice(monkeypatch, capsys):
    """An unreachable PyPI is advisory: --os still starts and prints no notice.

    ``check_for_update`` catches any failure -- no network, a missing
    distribution, or malformed JSON -- and reports "no update", so the
    "new version is available" banner must not appear and startup must reach
    the async entrypoint.
    """
    with _fresh_os_import(monkeypatch) as loop:
        monkeypatch.setattr(
            requests, "get", mock.Mock(side_effect=requests.ConnectionError)
        )

        with pytest.raises(SystemExit):
            importlib.import_module("interpreter")

    loop.run_async_main.assert_called_once_with()
    assert "new version" not in capsys.readouterr().out


def test_print_markdown_falls_back_when_a_rule_cannot_be_encoded(monkeypatch, capsys):
    """A console that cannot render rich's rule glyph still gets a divider.

    ``print_markdown`` renders a ``---`` line as a rich Rule; a legacy
    non-UTF-8 console raises UnicodeEncodeError for the glyph, which would
    otherwise abort the update banner mid-render. The fallback prints the
    plain-text divider instead.
    """
    modules = []

    with _fresh_os_import(monkeypatch) as loop:
        loop.run_async_main = mock.Mock(
            side_effect=lambda: modules.append(sys.modules["interpreter"])
        )
        response = mock.Mock()
        response.json.return_value = {"info": {"version": "0.0.0"}}
        monkeypatch.setattr(requests, "get", lambda *a, **k: response)

        with pytest.raises(SystemExit):
            importlib.import_module("interpreter")

    module = modules[0]

    def explode(*_args, **_kwargs):
        raise UnicodeEncodeError("ascii", "", 0, 1, "ordinal not in range(128)")

    monkeypatch.setattr(module, "rich_print", explode)
    module.print_markdown("---")

    assert capsys.readouterr().out == "---\n"
