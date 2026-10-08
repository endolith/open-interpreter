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
import re
from pathlib import Path


def _os_branch():
    """Return the `if "--os" in sys.argv:` branch of interpreter/__init__.py."""
    source = Path("interpreter/__init__.py").read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.If):
            return node
    raise AssertionError("no top-level if branch found in interpreter/__init__.py")


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
    """check_for_update compares with the packaging alias, not version.parse."""
    source = Path("interpreter/__init__.py").read_text()
    assert "packaging_version.parse(" in source
    assert re.search(r"(?<!packaging_)version\.parse\(", source) is None, (
        "check_for_update must not call version.parse (the shadowed name)"
    )


def _check_for_update_callable():
    """Build the real check_for_update() from the --os branch and call it.

    The function is defined at module level inside the branch, so it is not
    importable. Compile the branch's imports together with the function
    definition and exec them: that both supplies the names the function
    closes over and exercises the aliased packaging import itself, which no
    test can otherwise reach because the branch only runs under `--os`.
    """
    branch = _os_branch()
    # Everything up to the first `if` is the banner, the update check, and the
    # imports they need; past it sits the network call and the computer-use
    # bootstrap, which must not run here.
    body = []
    for node in branch.body:
        if isinstance(node, ast.If):
            break
        body.append(node)
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
    import requests

    class _Response:
        def json(self):
            return {"info": {"version": "9999.0.0"}}

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    assert _check_for_update_callable()() is True


def test_check_for_update_reports_no_update_when_current(monkeypatch):
    """A PyPI version equal to the installed one is not an update."""
    import importlib.metadata

    import requests

    class _Response:
        def json(self):
            return {"info": {"version": importlib.metadata.version("open-interpreter")}}

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    assert _check_for_update_callable()() is False
