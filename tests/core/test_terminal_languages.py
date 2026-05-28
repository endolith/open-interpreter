import os
import platform
import unittest
from unittest.mock import patch

from interpreter.core.terminal.base_language import format_execute_language_description
from interpreter.core.terminal.languages.resolve_bash import resolve_bash_executable
from interpreter.core.terminal.languages.applescript import AppleScript
from interpreter.core.terminal.languages.java import preprocess_java
from interpreter.core.terminal.languages.javascript import preprocess_javascript
from interpreter.core.terminal.languages.powershell import PowerShell, has_multiline_constructs
from interpreter.core.terminal.languages.r import R
from interpreter.core.terminal.languages.ruby import Ruby
from interpreter.core.terminal.languages.resolve_powershell import (
    powershell_startup_args,
    resolve_powershell_executable,
)
from interpreter.core.terminal.terminal import (
    _default_terminal_languages,
    _sync_active_line_detection_env,
)


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

    def test_sync_active_line_detection_env_follows_highlight_active_line(self):
        class FakeInterpreter:
            highlight_active_line = False

        _sync_active_line_detection_env(FakeInterpreter())
        self.assertEqual(os.environ["INTERPRETER_ACTIVE_LINE_DETECTION"], "false")

        FakeInterpreter.highlight_active_line = True
        _sync_active_line_detection_env(FakeInterpreter())
        self.assertEqual(os.environ["INTERPRETER_ACTIVE_LINE_DETECTION"], "true")

    def test_has_multiline_constructs_detects_hash_and_blocks(self):
        # Hash literal — the construct that caused the original parse error
        self.assertTrue(has_multiline_constructs("$h = @{\n    key = 'value'\n}"))
        # Script block / if / try bodies
        self.assertTrue(has_multiline_constructs("if ($x) {\n    Write-Host $x\n}"))
        self.assertTrue(has_multiline_constructs("try {\n    $x\n} catch {}"))
        # Pipeline continuation
        self.assertTrue(has_multiline_constructs("Get-Process |\n    Sort-Object CPU"))
        # Backtick continuation
        self.assertTrue(has_multiline_constructs("Get-Process `\n    -Name notepad"))
        # Here-string
        self.assertTrue(has_multiline_constructs('@"\nhello\n"@'))
        # Single-line code is not multiline
        self.assertFalse(has_multiline_constructs('$x = "hello"; Write-Host $x'))
        self.assertFalse(has_multiline_constructs("Get-Process"))

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

    def test_active_line_injection_disabled_when_env_false(self):
        """All language preprocessors respect INTERPRETER_ACTIVE_LINE_DETECTION=false."""
        with patch.dict("os.environ", {"INTERPRETER_ACTIVE_LINE_DETECTION": "false"}):
            # JavaScript
            js = preprocess_javascript("let x = 1;")
            self.assertNotIn("##active_line", js)

            # Ruby
            ruby = Ruby()
            rb = ruby.preprocess_code("x = 1")
            self.assertNotIn("##active_line", rb)

            # R
            r = R()
            rcode = r.preprocess_code("x <- 1")
            self.assertNotIn("##active_line", rcode)

            # Java (markers go inside the class body — just check raw preprocessor)
            java = preprocess_java("System.out.println(1);")
            self.assertNotIn("##active_line", java)

            # AppleScript
            aps = AppleScript()
            result = aps.add_active_line_indicators("do shell script \"echo hi\"")
            self.assertNotIn("##active_line", result)

            # PowerShell
            ps = PowerShell()
            pw = ps.preprocess_code("Write-Host 1")
            self.assertNotIn("##active_line", pw)

    def test_active_line_injection_present_when_env_true(self):
        """All language preprocessors inject markers when INTERPRETER_ACTIVE_LINE_DETECTION=true."""
        with patch.dict("os.environ", {"INTERPRETER_ACTIVE_LINE_DETECTION": "true"}):
            # JavaScript (only injected for single-line / non-multiline)
            js = preprocess_javascript("let x = 1;")
            self.assertIn("##active_line", js)

            # Ruby
            ruby = Ruby()
            rb = ruby.preprocess_code("x = 1")
            self.assertIn("##active_line", rb)

            # R
            r = R()
            rcode = r.preprocess_code("x <- 1")
            self.assertIn("##active_line", rcode)

            # Java
            java = preprocess_java("System.out.println(1);")
            self.assertIn("##active_line", java)

            # AppleScript
            aps = AppleScript()
            result = aps.add_active_line_indicators("do shell script \"echo hi\"")
            self.assertIn("##active_line", result)

            # PowerShell (single-line, no multiline constructs)
            ps = PowerShell()
            pw = ps.preprocess_code("Write-Host 1")
            self.assertIn("##active_line", pw)

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
