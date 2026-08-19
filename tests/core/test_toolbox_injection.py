import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from interpreter.core.core import OpenInterpreter
from interpreter.core.terminal.terminal import Terminal


class FakeToolbox:
    def __init__(self):
        self.import_toolbox_api = True
        self._has_imported_toolbox_api = False
        self.import_skills = False
        self._has_imported_skills = False
        self.verbose = False


class FakeInterpreter:
    def __init__(self):
        self.toolbox = FakeToolbox()


def _repo_root():
    # Git checkout root: tests/core/test_toolbox_injection.py -> three dirname()
    # levels up. Contains the `interpreter` package under test.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBuildImportToolboxApiCode(unittest.TestCase):
    def test_code_prepends_parent_interpreter_package_dir_to_sys_path(self):
        """The injection must force the kernel to import the exact same
        `interpreter` package the parent CLI process is running, even when the
        kernel's working directory contains a different copy of the package."""
        terminal = Terminal(OpenInterpreter.__new__(OpenInterpreter))
        code = terminal._build_import_toolbox_api_code()
        self.assertIn(repr(_repo_root()), code)
        self.assertIn("sys.path.insert(0, _pkg_dir)", code)

    def test_code_contains_success_sentinel_and_core_imports(self):
        """The injection must still define `toolbox` and `ai2`, and print a
        sentinel the caller uses to confirm the injection succeeded."""
        terminal = Terminal(OpenInterpreter.__new__(OpenInterpreter))
        code = terminal._build_import_toolbox_api_code()
        self.assertIn("from interpreter import interpreter, ai2", code)
        self.assertIn("toolbox = interpreter.toolbox", code)
        self.assertIn("__TOOLBOX_API_IMPORTED__", code)
        self.assertIn('os.environ["INTERPRETER_TOOLBOX_API"] = "False"', code)


class TestToolboxInjectionGate(unittest.TestCase):
    def _terminal(self):
        return Terminal(FakeInterpreter())

    def test_flag_restored_when_injection_fails(self):
        """If the injection does not succeed (e.g. the kernel imported the
        wrong copy of the package and `ai2`/`toolbox` don't exist), the
        `_has_imported_toolbox_api` flag must be restored to False so the next
        toolbox use retries instead of surfacing a confusing
        `NameError: name 'toolbox' is not defined`."""
        terminal = self._terminal()
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("INTERPRETER_TOOLBOX_API", None)
            with patch.object(
                Terminal,
                "_streaming_run",
                return_value=[
                    {
                        "type": "console",
                        "format": "output",
                        "content": "ImportError: cannot import name 'ai2'",
                    }
                ],
            ) as mock_run:
                terminal.run("python", "toolbox.mouse.position()")
        self.assertFalse(terminal.interpreter.toolbox._has_imported_toolbox_api)
        # The injection must have been attempted (as a recursive run) with the
        # real injection code, followed by the user's own code.
        codes = [call.args[1] for call in mock_run.call_args_list]
        self.assertEqual(len(codes), 2)
        self.assertIn("__TOOLBOX_API_IMPORTED__", codes[0])
        self.assertEqual(codes[1], "toolbox.mouse.position()")

    def test_flag_kept_when_injection_succeeds(self):
        """When the injection succeeds (sentinel printed), the flag stays True
        so the API is only injected once per session."""
        terminal = self._terminal()
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("INTERPRETER_TOOLBOX_API", None)
            with patch.object(
                Terminal,
                "_streaming_run",
                return_value=[
                    {
                        "type": "console",
                        "format": "output",
                        "content": "__TOOLBOX_API_IMPORTED__\n",
                    }
                ],
            ):
                terminal.run("python", "toolbox.mouse.position()")
        self.assertTrue(terminal.interpreter.toolbox._has_imported_toolbox_api)


class TestInjectionResistsShadowingCheckout(unittest.TestCase):
    def test_injection_imports_parent_package_despite_shadowing_cwd(self):
        """Regression test: launching the CLI from inside a *different*
        open-interpreter checkout (e.g. the `main`-branch repo, whose
        `interpreter` package has no `toolbox` and no `ai2`) used to make the
        kernel's `from interpreter import interpreter, ai2` import that wrong
        copy, so the toolbox API never landed and every toolbox use raised
        `NameError: name 'toolbox' is not defined`. The generated injection
        code must instead resolve to the parent process's own package.

        The control below is the pre-fix injection body: run with a working
        directory that shadows the real package, it fails. The generated code
        from `_build_import_toolbox_api_code` must succeed in the same
        situation.
        """
        terminal = Terminal(OpenInterpreter.__new__(OpenInterpreter))
        injected_code = terminal._build_import_toolbox_api_code()
        control_code = textwrap.dedent(
            """
            import os
            os.environ["INTERPRETER_TOOLBOX_API"] = "False"
            import time
            import datetime
            from interpreter import interpreter, ai2
            toolbox = interpreter.toolbox
            print("__TOOLBOX_API_IMPORTED__")
            """
        ).strip()

        with tempfile.TemporaryDirectory() as shadow_dir:
            # A decoy `interpreter` package like the main-branch checkout:
            # has `interpreter` but neither `toolbox` nor `ai2`.
            decoy = os.path.join(shadow_dir, "interpreter")
            os.makedirs(decoy)
            with open(os.path.join(decoy, "__init__.py"), "w") as f:
                f.write(
                    "class _Decoy:\n"
                    "    pass\n"
                    "interpreter = _Decoy()\n"
                    "# deliberately no `toolbox` and no `ai2`\n"
                )

            env = dict(os.environ)
            # Make the real package importable from anywhere, as in an
            # editable install, while the shadowing cwd still takes precedence
            # at sys.path[0] unless the injection inserts its own entry first.
            env["PYTHONPATH"] = _repo_root() + os.pathsep + env.get("PYTHONPATH", "")

            control = subprocess.run(
                [sys.executable, "-c", control_code],
                cwd=shadow_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertNotIn(
                "__TOOLBOX_API_IMPORTED__",
                control.stdout,
                "control (pre-fix) injection unexpectedly imported the real package",
            )

            fixed = subprocess.run(
                [sys.executable, "-c", injected_code],
                cwd=shadow_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertIn(
                "__TOOLBOX_API_IMPORTED__",
                fixed.stdout,
                f"injection failed against shadowing checkout:\n{fixed.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
