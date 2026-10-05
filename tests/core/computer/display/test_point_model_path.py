import ast
import os
import unittest


def _point_module_path():
    import interpreter

    return os.path.join(
        os.path.dirname(interpreter.__file__),
        "core", "toolbox", "display", "point", "point.py",
    )


def _parse_point_module():
    path = _point_module_path()
    if not os.path.isfile(path):
        raise AssertionError(f"cannot find {path}")
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read())


def _module_level_assignments(tree):
    """Names assigned by module-level `name = ...` statements."""
    return {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


class TestPointModelPathIsDefined(unittest.TestCase):
    """The grounding model's cache path must exist as a name in the module.

    model_path was read three times and never assigned, so the block that uses
    it raised NameError on its first os.path.isfile call. Nothing caught this
    because `fast_model = True` at module level means the block never runs, so
    the bug sits latent behind a flag nobody flips -- and it surfaces the first
    time anyone does.

    Checked statically rather than by import: this module needs cv2, timm, torch
    and sentence_transformers, none of which CI installs, so a runtime test
    would skip and prove nothing. Reading the AST needs no dependencies and
    cannot skip.
    """

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

        oi_dir is imported at the top of the module and used nowhere else, which
        is what identified the intended location. Caching under a bare relative
        path would instead depend on the working directory.
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
