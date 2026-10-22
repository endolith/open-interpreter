"""The point module's grounding-model cache path must be a defined name.

``model_path`` was read three times in ``point.py`` and assigned nowhere, so
the block that uses it raised ``NameError`` on its first ``os.path.isfile``
call. Nothing caught it because ``fast_model = True`` at module level makes
``if fast_model == False`` false, so the block never runs — the bug sits
latent behind a flag nobody flips and surfaces the first time anyone does.

Checked statically rather than by import: this module needs cv2, timm, torch
and sentence_transformers, none of which CI installs, so a runtime test would
skip and prove nothing. Reading the AST needs no dependencies and cannot skip.
"""

import ast
import os
import unittest


def _point_module_path():
    """Absolute path to ``point.py`` in this checkout."""
    import interpreter

    return os.path.join(
        os.path.dirname(interpreter.__file__),
        "core",
        "computer",
        "display",
        "point",
        "point.py",
    )


def _parse_point_module():
    """Parse ``point.py`` without importing it (its deps are optional)."""
    path = _point_module_path()
    if not os.path.isfile(path):
        raise AssertionError(f"cannot find {path}")
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read())


def _module_level_assignments(tree):
    """Names assigned by module-level ``name = ...`` statements."""
    return {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


class TestPointModelPathIsDefined(unittest.TestCase):
    """The grounding model's cache path must exist as a name in the module."""

    def test_model_path_is_assigned_at_module_level(self):
        """A module-level assignment to model_path exists."""
        assigned = _module_level_assignments(_parse_point_module())
        self.assertIn(
            "model_path",
            assigned,
            "model_path is read three times below but never assigned; flipping "
            "fast_model to False would raise NameError",
        )

    def test_model_path_is_built_from_the_oi_config_dir(self):
        """The path is derived from oi_dir, the repo's per-user location.

        Caching under a bare relative path would instead depend on the working
        directory, which is not where the rest of OI keeps per-user state.
        """
        tree = _parse_point_module()
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "model_path"
                for t in node.targets
            ):
                continue
            names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            self.assertIn(
                "oi_dir",
                names,
                "model_path should be built from oi_dir, the unused import in "
                "this module and the convention for per-user state",
            )
            return
        self.fail("no module-level assignment to model_path")

    def test_cache_directory_is_not_created_at_import(self):
        """Importing point must not create oi_dir before the save branch.

        os.makedirs(oi_dir) ran as a module-level statement, so an unwritable
        location raised at import even when fast_model was True and the
        grounding-model branch never ran. It now runs inside that branch, just
        before the model is saved.
        """
        for stmt in _parse_point_module().body:
            if isinstance(
                stmt, (ast.If, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                self.assertFalse(
                    isinstance(func, ast.Attribute) and func.attr == "makedirs",
                    "os.makedirs must not run unconditionally at import time",
                )


    def test_cache_directory_is_created_in_the_save_branch(self):
        """The save branch creates oi_dir before writing the model.

        The line is excluded from coverage because the grounding model needs
        torch/timm, which CI does not install; this AST check keeps the
        behaviour verified without importing those dependencies.
        """
        for node in ast.walk(_parse_point_module()):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
                continue
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "makedirs"
                for call in ast.walk(node)
            ):
                return
        self.fail("the model-save branch does not create oi_dir")
