import platform
import unittest

from interpreter.core.terminal.base_language import format_execute_language_description
from interpreter.core.terminal.languages.resolve_bash import resolve_bash_executable
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
