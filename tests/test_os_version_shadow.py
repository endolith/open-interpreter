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
    assert "packaging" in (bound.get("packaging_version") or ""), (
        "packaging.version must be imported under an alias"
    )


def test_os_update_check_parses_with_packaging_alias():
    """check_for_update compares with the packaging alias, not version.parse."""
    source = Path("interpreter/__init__.py").read_text()
    assert "packaging_version.parse(" in source
    assert re.search(r"(?<!packaging_)version\.parse\(", source) is None, (
        "check_for_update must not call version.parse (the shadowed name)"
    )
