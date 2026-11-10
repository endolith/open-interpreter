"""Regression test for #407: --os startup shadowing.

interpreter/__init__.py once bound `version` twice inside the --os
branch — first to importlib.metadata.version (a function), then to
packaging.version (a module) — so check_for_update() called a module
and every `interpreter --os` start died at import with TypeError.

Importing that branch has heavy side effects (network check, async
loop, exit), so this pins the fix structurally: `version` must resolve
to the importlib function while the packaging import lives under an
alias, and the comparison must use the alias.
"""

import ast
from pathlib import Path


def _os_branch():
    """Return the `if "--os" in sys.argv:` branch of interpreter/__init__.py."""
    source = Path("interpreter/__init__.py").read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.If) and "--os" in ast.unparse(node.test):
            return node
    raise AssertionError("no top-level --os branch found in interpreter/__init__.py")


def test_os_branch_version_imports_do_not_collide():
    """version stays the importlib function; packaging uses an alias."""
    branch = _os_branch()
    bound = {}
    for node in ast.walk(branch):
        if isinstance(node, ast.ImportFrom):
            for name in node.names:
                bound[name.asname or name.name] = node.module

    assert bound.get("version") == "importlib.metadata", (
        f"version must come from importlib.metadata, got {bound.get('version')}"
    )
    assert "packaging" in (bound.get("packaging_version") or ""), "packaging.version must be imported under an alias"


def test_os_update_check_parses_with_packaging_alias():
    """check_for_update compares with the packaging alias, not version.parse.

    Checked on call nodes rather than with a regex: a substring guard also
    matches `packaging_version.parse(`, comments, and docstrings.
    """
    parse_calls = [
        node
        for node in ast.walk(_os_branch())
        if isinstance(node, ast.Attribute) and node.attr == "parse"
    ]
    shadowed = [
        call
        for call in parse_calls
        if isinstance(call.value, ast.Name) and call.value.id == "version"
    ]
    aliased = [
        call
        for call in parse_calls
        if isinstance(call.value, ast.Name) and call.value.id == "packaging_version"
    ]

    assert not shadowed, "check_for_update must not call version.parse (the shadowed name)"
    assert aliased, "check_for_update should parse with the packaging_version alias"


def _check_for_update_callable():
    """Build the real check_for_update() from the --os branch and call it.

    The function is defined at module level inside the branch, so it is not
    importable. Compile the branch's imports together with the function
    definition and exec them: that both supplies the names the function
    closes over and exercises the aliased packaging import itself, which no
    test can otherwise reach because the branch only runs under `--os`.
    """
    branch = _os_branch()
    # Only the imports the function closes over and the function itself: the
    # banner, the network call and the computer-use bootstrap must not run here.
    body = [
        node
        for node in branch.body
        if isinstance(node, ast.Import)
        # Absolute imports supply the names check_for_update closes over; the
        # relative computer-use import is what a bare exec cannot resolve.
        or (isinstance(node, ast.ImportFrom) and node.level == 0)
        or (isinstance(node, ast.FunctionDef) and node.name == "check_for_update")
    ]
    module = ast.Module(body=body, type_ignores=[])
    namespace = {"__name__": "interpreter"}
    exec(compile(module, "interpreter/__init__.py", "exec"), namespace)
    return namespace["check_for_update"]


def test_check_for_update_detects_newer_release(monkeypatch):
    """check_for_update() returns True when PyPI advertises a newer version (issue #407).

    This is the call that used to die with TypeError: `version` had been
    rebound to the packaging module, so `version.parse` raised before the
    comparison could run.
    """
    import importlib.metadata

    import requests

    class _Response:
        def json(self):
            return {"info": {"version": "9999.0.0"}}

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    # Pin the installed version: the test must not depend on this checkout
    # being pip-installed as `open-interpreter`.
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.1")
    assert _check_for_update_callable()() is True


def test_check_for_update_survives_a_failed_pypi_request(monkeypatch):
    """A failed PyPI request reports "no update" instead of aborting --os.

    The notice is advisory: the request used to run unguarded, so a network
    failure at import time escaped and stopped the --os branch before its
    handoff.
    """
    import requests

    def explode(*args, **kwargs):
        raise requests.exceptions.ConnectionError("PyPI unreachable")

    monkeypatch.setattr(requests, "get", explode)
    assert _check_for_update_callable()() is False


def test_check_for_update_reports_no_update_when_current(monkeypatch):
    """A PyPI version equal to the installed one is not an update."""
    import importlib.metadata

    import requests

    class _Response:
        def json(self):
            return {"info": {"version": "0.0.5"}}

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.5")
    assert _check_for_update_callable()() is False
