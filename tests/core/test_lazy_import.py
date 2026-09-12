import os
import shutil
import sys
import tempfile
import types
import unittest

from interpreter.core.utils.lazy_import import lazy_import


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


if __name__ == "__main__":
    unittest.main()
