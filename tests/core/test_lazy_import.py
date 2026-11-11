import importlib
import os
import shutil
import sys
import tempfile
import types
import unittest

from interpreter.core.utils.lazy_import import (
    lazy_import,
    lazy_import_attribute,
    lazy_import_deferred,
)


class TestLazyImport(unittest.TestCase):
    def setUp(self):
        """Create an isolated directory for fake importable modules."""
        self.tmp = tempfile.mkdtemp(prefix="oi_lazy_test_")
        sys.path.insert(0, self.tmp)
        self._added_modules = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        """Remove the temp dir from sys.path and purge test modules."""
        sys.path.remove(self.tmp)
        shutil.rmtree(self.tmp, ignore_errors=True)
        for name in self._added_modules:
            sys.modules.pop(name, None)

    def _write_module(self, name, code):
        """Write a single-file module into the temp dir and track it for cleanup."""
        with open(os.path.join(self.tmp, name + ".py"), "w") as f:
            f.write(code)
        self._added_modules.append(name)

    def _write_package(self, name, submodule, submodule_code):
        """Write a package with one submodule into the temp dir and track it."""
        package_dir = os.path.join(self.tmp, name)
        os.makedirs(package_dir, exist_ok=True)
        with open(os.path.join(package_dir, "__init__.py"), "w") as f:
            f.write("")
        with open(os.path.join(package_dir, submodule + ".py"), "w") as f:
            f.write(submodule_code)
        self._added_modules.extend([name, f"{name}.{submodule}"])

    def test_missing_optional_returns_none(self):
        """Verify a nonexistent optional module still returns None without raising."""
        self.assertIsNone(lazy_import("oi_no_such_module_xyz_123"))

    def test_broken_module_introspection_safe(self):
        """Verify touching __file__ etc. never executes a module whose import fails."""
        self._write_module("oi_broken_mod", "raise ImportError('no display here')\n")
        mod = lazy_import("oi_broken_mod")
        # Introspection must not trigger (or choke on) the real import.
        self.assertTrue(hasattr(mod, "__file__"))
        self.assertTrue(mod.__file__.endswith("oi_broken_mod.py"))
        self.assertEqual(mod.__name__, "oi_broken_mod")
        # Placeholder retained, not a poisoned half-imported module.
        self.assertIs(sys.modules["oi_broken_mod"], mod)

    def test_broken_module_real_use_raises(self):
        """Verify real attribute use raises the original import error, every time."""
        self._write_module("oi_broken_use", "raise ImportError('no display here')\n")
        mod = lazy_import("oi_broken_use")
        with self.assertRaises(ImportError) as context:
            mod.anything
        self.assertIn("no display", str(context.exception))
        with self.assertRaises(ImportError):
            mod.anything  # Cached failure re-raises without re-executing.
        self.assertIs(sys.modules["oi_broken_use"], mod)

    def test_working_module_loads_on_use(self):
        """Verify a healthy module loads transparently and swaps into sys.modules."""
        self._write_module("oi_good_mod", "VALUE = 42\n")
        mod = lazy_import("oi_good_mod")
        self.assertEqual(mod.VALUE, 42)
        real = sys.modules["oi_good_mod"]
        self.assertIsNot(real, mod)
        self.assertIs(type(real), types.ModuleType)
        self.assertEqual(real.VALUE, 42)

    def test_setattr_writes_through_to_the_real_module(self):
        """Verify assignments configure the real module, not the placeholder.

        OI sets `litellm.suppress_debug_info` and similar module attributes at
        call time. If those writes landed on the placeholder, the real module
        would never see them, and dict-first reads would later mask the real
        module's value.
        """
        self._write_module("oi_settable_mod", "FLAG = False\n")
        mod = lazy_import("oi_settable_mod")
        mod.FLAG = True
        self.assertTrue(mod.FLAG)
        real = sys.modules["oi_settable_mod"]
        self.assertIsNot(real, mod)
        self.assertTrue(real.FLAG)

    def test_delattr_removes_the_attribute_from_the_real_module(self):
        """Verify monkeypatch.delattr on a lazy module reaches the real module."""
        self._write_module("oi_deletable_mod", "VALUE = 1\n")
        mod = lazy_import("oi_deletable_mod")
        self.assertEqual(mod.VALUE, 1)
        del mod.VALUE
        self.assertFalse(hasattr(mod, "VALUE"))
        real = sys.modules["oi_deletable_mod"]
        self.assertIsNot(real, mod)
        self.assertFalse(hasattr(real, "VALUE"))

    def test_deferred_module_defers_dotted_import_until_use(self):
        """Verify creating a deferred proxy imports neither the parent nor the submodule.

        find_spec on a dotted name imports its parent package, which is the
        cost the deferred proxy exists to avoid: the parent must not appear in
        sys.modules until the proxy is actually used.
        """
        self._write_package("oi_deferred_pkg", "sub", "VALUE = 7\n")
        mod = lazy_import_deferred("oi_deferred_pkg.sub")
        self.assertNotIn("oi_deferred_pkg", sys.modules)
        self.assertNotIn("oi_deferred_pkg.sub", sys.modules)

        self.assertEqual(mod.VALUE, 7)
        self.assertEqual(sys.modules["oi_deferred_pkg.sub"].VALUE, 7)

    def test_deferred_module_returns_already_imported_module(self):
        """Verify an already-imported module is returned directly, not wrapped."""
        self._write_module("oi_deferred_loaded", "VALUE = 3\n")
        real = importlib.import_module("oi_deferred_loaded")
        self.assertIs(lazy_import_deferred("oi_deferred_loaded"), real)

    def test_deferred_attribute_calls_after_importing(self):
        """Verify a deferred attribute stays unimported until called, then delegates."""
        self._write_package("oi_attr_pkg", "sub", "def value():\n    return 5\n")
        attr = lazy_import_attribute("oi_attr_pkg.sub", "value")
        self.assertNotIn("oi_attr_pkg", sys.modules)

        self.assertEqual(attr(), 5)
        self.assertIn("oi_attr_pkg.sub", sys.modules)

    def test_deferred_attribute_forwards_attribute_access(self):
        """Verify attribute reads on the proxy reach the resolved object."""
        self._write_package("oi_attr_pkg2", "sub", "class Obj:\n    SIZE = 9\n\nobj = Obj()\n")
        attr = lazy_import_attribute("oi_attr_pkg2.sub", "obj")
        self.assertEqual(attr.SIZE, 9)


if __name__ == "__main__":
    unittest.main()
