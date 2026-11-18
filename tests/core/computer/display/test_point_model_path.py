import ast
import os
import unittest


def _point_module_path():
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

    `model_path` was read three times and never assigned, so the block that uses
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

def _oi_dir_makedirs_call_lines(tree):
    """Line numbers of `os.makedirs(oi_dir, ...)` calls in the module."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "makedirs"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "oi_dir"
    ]


def _slow_path_block(tree):
    """The module-level `if fast_model == False:` node."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "fast_model"
        ):
            return node
    raise AssertionError("no module-level `if fast_model == False:` block")


class TestCacheDirectoryIsCreatedOnTheSlowPathOnly(unittest.TestCase):
    """Importing the module must not create the grounding cache directory.

    `os.makedirs(oi_dir, exist_ok=True)` ran at import, before the fast_model
    flag was consulted, so a read-only home raised PermissionError and the
    module failed to import even though the default fast path never reads or
    writes the cache. The directory belongs next to the torch.save call that
    actually needs it.

    Checked statically for the same reason as the model_path tests: importing
    this module needs cv2, timm, torch and sentence_transformers.
    """

    def test_makedirs_is_only_called_inside_the_slow_path(self):
        """Every cache-directory creation sits inside the fast_model block."""
        tree = _parse_point_module()
        block = _slow_path_block(tree)
        lines = _oi_dir_makedirs_call_lines(tree)
        self.assertTrue(lines, "the slow path should create its cache directory")
        for line in lines:
            self.assertTrue(
                block.lineno < line <= block.end_lineno,
                f"os.makedirs at line {line} runs outside the fast_model block",
            )

    def test_cache_directory_is_created_before_the_model_is_saved(self):
        """A created directory precedes torch.save, so the write cannot fail."""
        tree = _parse_point_module()
        block = _slow_path_block(tree)
        save_line = min(
            node.lineno
            for node in ast.walk(block)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "save"
        )
        self.assertTrue(
            any(
                node.lineno < save_line
                for node in ast.walk(block)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "makedirs"
            ),
            "torch.save would fail when the cache directory does not exist",
        )

