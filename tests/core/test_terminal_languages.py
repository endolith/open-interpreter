import os
import platform
import unittest
from unittest.mock import patch

from interpreter.core.terminal.base_language import format_execute_language_description
from interpreter.core.terminal.languages.resolve_bash import resolve_bash_executable
from interpreter.core.terminal.languages.powershell import PowerShell
from interpreter.core.terminal.languages.resolve_powershell import (
    powershell_startup_args,
    resolve_powershell_executable,
)
from interpreter.core.terminal.terminal import _default_terminal_languages


class TestTerminalLanguages(unittest.TestCase):
    def test_shell_language_removed(self):
        names = {lang.name.lower() for lang in _default_terminal_languages()}
        self.assertNotIn("shell", names)
        self.assertIn("bash", names)
        self.assertIn("perl", names)
        self.assertIn("augeas", names)
        if platform.system() == "Windows":
            self.assertIn("cmd", names)
        else:
            self.assertNotIn("cmd", names)

    def test_powershell_startup_args_loads_profile_with_process_bypass(self):
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("INTERPRETER_POWERSHELL_NO_PROFILE", None)
            args = powershell_startup_args()
        self.assertIn("-NoLogo", args)
        self.assertIn("-ExecutionPolicy", args)
        self.assertEqual(args[args.index("-ExecutionPolicy") + 1], "Bypass")
        self.assertNotIn("-NoProfile", args)

    def test_powershell_startup_args_no_profile_env(self):
        with patch.dict("os.environ", {"INTERPRETER_POWERSHELL_NO_PROFILE": "1"}):
            args = powershell_startup_args()
        self.assertIn("-NoProfile", args)

    def test_powershell_line_postprocessor_filters_prompt_and_continuation(self):
        ps = PowerShell()
        # PS prompt lines are suppressed (with and without conda prefix)
        self.assertIsNone(ps.line_postprocessor("PS C:\\Users\\Jonathan> "))
        self.assertIsNone(ps.line_postprocessor("(base) PS C:\\Users\\Jonathan> try {"))
        self.assertIsNone(ps.line_postprocessor("PS D:\\work> "))
        # Continuation-prompt echo lines are suppressed
        self.assertIsNone(ps.line_postprocessor(">>"))
        self.assertIsNone(ps.line_postprocessor(">>     $ErrorActionPreference = 'Stop'"))
        self.assertIsNone(ps.line_postprocessor(">> Write-Host hello"))
        # Real output is kept
        self.assertEqual(
            ps.line_postprocessor("Hello from PowerShell!"), "Hello from PowerShell!"
        )
        self.assertEqual(ps.line_postprocessor("True"), "True")
        self.assertEqual(ps.line_postprocessor("42"), "42")
        # "PS C:\" embedded mid-line (not a prompt) is NOT filtered
        self.assertEqual(
            ps.line_postprocessor("Path is PS C:\\Users\\foo"),
            "Path is PS C:\\Users\\foo",
        )

    def test_resolve_bash_executable(self):
        path = resolve_bash_executable()
        self.assertTrue(path.endswith("bash") or path.endswith("bash.exe"))

    def test_get_language_rejects_aliases(self):
        from interpreter import interpreter

        terminal = interpreter.terminal
        self.assertIsNotNone(terminal.get_language("bash"))
        self.assertIsNone(terminal.get_language("sh"))
        self.assertIsNone(terminal.get_language("py"))

    @unittest.skipUnless(platform.system() != "Windows", "Unix only")
    def test_powershell_fails_without_pwsh(self):
        with patch(
            "interpreter.core.terminal.languages.resolve_powershell.shutil.which",
            return_value=None,
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_powershell_executable()
            self.assertNotIn("bash", str(ctx.exception).lower())

    def test_format_execute_language_description_includes_notes(self):
        desc = format_execute_language_description(_default_terminal_languages())
        self.assertIn("bash", desc)
        self.assertIn("Language notes:", desc)
        self.assertIn("GNU bash", desc)
        if platform.system() == "Windows":
            self.assertIn("cmd", desc)
            self.assertIn("cmd.exe", desc)


if __name__ == "__main__":
    unittest.main()
