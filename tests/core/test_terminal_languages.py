import platform
import unittest
from unittest.mock import patch

from interpreter.core.terminal.base_language import format_execute_language_description
from interpreter.core.terminal.languages.resolve_bash import resolve_bash_executable
from interpreter.core.terminal.languages.resolve_powershell import resolve_powershell_executable
from interpreter.core.terminal.terminal import _default_terminal_languages


class TestTerminalLanguages(unittest.TestCase):
    def test_shell_language_removed(self):
        names = {lang.name.lower() for lang in _default_terminal_languages()}
        self.assertNotIn("shell", names)
        self.assertIn("bash", names)
        if platform.system() == "Windows":
            self.assertIn("cmd", names)
        else:
            self.assertNotIn("cmd", names)

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
