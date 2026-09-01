import os
import platform
import tempfile
import unittest
from unittest.mock import patch

from interpreter.core.terminal.base_language import format_execute_language_description
from interpreter.core.terminal.languages.bash import Bash
from interpreter.core.terminal.languages.cwd_tracking import CwdTrackingMixin
from interpreter.core.terminal.languages.jupyter_language import (
    JupyterLanguage,
    strip_redundant_imports,
)
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


class _StubCwdShell(CwdTrackingMixin):
    """CwdTrackingMixin without launching a real shell, for config-specific tests."""

    def __init__(self, **config):
        CwdTrackingMixin.__init__(self)
        for key, value in config.items():
            setattr(self, key, value)

    def _cwd_marker_echo(self):
        return ""


class _FakeSubprocess:
    """Duck-types SubprocessLanguage's init contract for mixin MRO tests."""

    def __init__(self):
        self.process = None
        self.start_cmd = []
        self.output_queue = []
        self.done = None


class _MixTest(CwdTrackingMixin, _FakeSubprocess):
    """Mirrors the Bash(CwdTrackingMixin, SubprocessLanguage) MRO."""

    def __init__(self):
        CwdTrackingMixin.__init__(self)
        _FakeSubprocess.__init__(self)


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
        self.assertIsNone(ps.line_postprocessor("PS C:\\Users\\user> "))
        self.assertIsNone(ps.line_postprocessor("(base) PS C:\\Users\\user> try {"))
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

    def test_bash_resolve_bash_executable(self):
        path = resolve_bash_executable()
        self.assertTrue(path.endswith("bash") or path.endswith("bash.exe"))

    def test_bash_init_initializes_subprocess_state(self):
        """Bash() must set the SubprocessLanguage attributes run() relies on (process, start_cmd, output_queue, done)."""
        bash = Bash()
        self.assertIsNone(bash.process)
        self.assertIsNotNone(bash.output_queue)
        self.assertIsNotNone(bash.done)
        self.assertTrue(bash.start_cmd)

    def test_mixin_mro_initializes_both_parents(self):
        """A CwdTrackingMixin+SubprocessLanguage MRO must run SubprocessLanguage.__init__ (regression: super() skipped it)."""
        mix = _MixTest()
        self.assertEqual(mix.cwd, os.getcwd())
        self.assertIsNone(mix.process)
        self.assertEqual(mix.start_cmd, [])

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

    def test_strip_redundant_imports_removes_plain_import_when_already_imported(self):
        """A top-level `import os` is dropped when `os` is already bound in the kernel namespace."""
        stripped, removed = strip_redundant_imports("import os\nos.getcwd()", {"os"})
        self.assertEqual(stripped, "os.getcwd()")
        self.assertEqual(removed, ["os"])

    def test_strip_redundant_imports_keeps_first_import(self):
        """The first `import os` is kept when `os` isn't tracked yet — stripping would break it."""
        code = "import os\nos.getcwd()"
        stripped, removed = strip_redundant_imports(code, set())
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_multiple_names_all_redundant(self):
        """A multi-name import line is dropped only when every name is allowlisted and already imported."""
        stripped, removed = strip_redundant_imports("import os, sys", {"os", "sys"})
        self.assertEqual(stripped, "")
        self.assertEqual(sorted(removed), ["os", "sys"])
        stripped, removed = strip_redundant_imports("import os, numpy", {"os", "sys"})
        self.assertEqual(stripped, "import os, numpy")
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_leaves_non_allowlisted_modules(self):
        """Imports with side effects (matplotlib) are never stripped even if already imported."""
        code = "import matplotlib.pyplot as plt\nimport matplotlib"
        stripped, removed = strip_redundant_imports(code, {"matplotlib"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_keeps_aliased_third_party_imports(self):
        """The universal aliased imports (numpy/pandas) are not stripped — they have aliases."""
        code = "import numpy as np\nimport pandas as pd"
        stripped, removed = strip_redundant_imports(code, {"numpy", "pandas"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_strips_common_boilerplate(self):
        """Common side-effect-free boilerplate (re, json, requests) is stripped when already imported."""
        stripped, removed = strip_redundant_imports(
            "import re\nimport json\nimport requests", {"re", "json", "requests"}
        )
        self.assertEqual(stripped, "")
        self.assertEqual(sorted(removed), ["json", "re", "requests"])

    def test_strip_redundant_imports_keeps_non_allowlisted_plain_import(self):
        """A plain import of a non-allowlisted module is kept even if already imported."""
        code = "import plotly"
        stripped, removed = strip_redundant_imports(code, {"plotly"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_leaves_alias_dotted_and_from_imports(self):
        """Aliased, dotted and `from ... import` lines are never stripped — removing them could break the code."""
        code = "import os as o\nimport os.path\nfrom os import getcwd"
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_ignores_indented_scoped_imports(self):
        """An `import` inside a function body (indented) is not stripped even if the module is imported."""
        code = "def f():\n    import os\n    return os.getcwd()"
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_handles_trailing_comments(self):
        """A comment after the import doesn't prevent stripping the redundant plain import."""
        stripped, removed = strip_redundant_imports("import os  # needed", {"os"})
        self.assertEqual(stripped, "")
        self.assertEqual(removed, ["os"])

    def test_strip_redundant_imports_keeps_import_after_statement(self):
        """An import below an executable statement is never stripped — `os = "string"` would be re-bound by the stripped import (regression)."""
        code = 'os = "string"\nimport os\nos.something()'
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_strips_leading_import_after_comment(self):
        """Comments/blank lines may precede the leading import block, which is still stripped."""
        code = "# first\n# second\n\nimport os\nos.getcwd()"
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, "# first\n# second\n\nos.getcwd()")
        self.assertEqual(removed, ["os"])

    def test_strip_redundant_imports_keeps_middle_import_after_statement_and_comment(self):
        """An import following a statement (even past a comment) is kept — only the leading block is eligible."""
        code = "x = 1\n# note\nimport os\nos.getcwd()"
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_shebang_counts_as_leading_comment(self):
        """A shebang is comment-like, so an import below it is still in the leading block and stripped."""
        code = "#!/usr/bin/env python\nimport os\nos.getcwd()"
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, "#!/usr/bin/env python\nos.getcwd()")
        self.assertEqual(removed, ["os"])

    def test_strip_redundant_imports_future_import_stays_in_leading_block(self):
        """A `from __future__ import` is import-like, so a later plain import is still leading and stripped."""
        code = "from __future__ import annotations\nimport os\nx: int = 1"
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, "from __future__ import annotations\nx: int = 1")
        self.assertEqual(removed, ["os"])

    def test_strip_redundant_imports_keeps_import_after_code_block(self):
        """An import below a suite (`if True: pass`) is not in the leading block and is kept."""
        code = "if True:\n    pass\n\nimport os\nos.getcwd()"
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_keeps_import_inside_triple_quote_string(self):
        """An `import` that only appears inside a string literal must never be touched."""
        code = 's = """\nimport os\n"""\nprint(s)'
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_keeps_docstring_then_import(self):
        """A module docstring is an expression statement, so an import below it is kept."""
        code = '"""doc"""\nimport os\nos.getcwd()'
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_keeps_trailing_comma_import(self):
        """`import os,` is malformed Python — never stripped, never crashes the stripper."""
        code = "import os,\nprint(1)"
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_is_case_sensitive(self):
        """`import OS` binds a different name than `os` — never stripped for tracked `os`."""
        stripped, removed = strip_redundant_imports("import OS\nprint(1)", {"os"})
        self.assertEqual(stripped, "import OS\nprint(1)")
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_strips_tab_separated_import(self):
        """`import\tos` is valid Python and, being allowlisted and imported, is stripped."""
        stripped, removed = strip_redundant_imports("import\tos\nprint(1)", {"os"})
        self.assertEqual(stripped, "print(1)")
        self.assertEqual(removed, ["os"])

    def test_strip_redundant_imports_keeps_parenthesized_multiline_import(self):
        """Parenthesized multi-line `import (...)` syntax is never matched by the line regex and is kept."""
        code = "import (\n    os,\n    sys,\n)\nprint(1)"
        stripped, removed = strip_redundant_imports(code, {"os", "sys"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_strip_redundant_imports_mixed_redundant_and_nonredundant_leading(self):
        """In a mixed leading block only the truly redundant plain import is dropped."""
        code = "import os\nimport numpy as np\nimport os.path\nprint(1)"
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(
            stripped, "import numpy as np\nimport os.path\nprint(1)"
        )
        self.assertEqual(removed, ["os"])

    def test_strip_redundant_imports_keeps_try_except_import(self):
        """`import` inside a try/except is guarded, hence never stripped."""
        code = "try:\n    import os\nexcept ImportError:\n    pass"
        stripped, removed = strip_redundant_imports(code, {"os"})
        self.assertEqual(stripped, code)
        self.assertEqual(removed, [])

    def test_jupyter_strip_boilerplate_returns_stripped_code_and_notice(self):
        """strip_boilerplate returns the code minus redundant imports plus a short notice."""
        jl = object.__new__(JupyterLanguage)  # skip kernel startup
        jl.imported_modules = {"os", "sys"}
        stripped, notice = jl.strip_boilerplate("import os\nimport sys\nos.getcwd()")
        self.assertNotIn("import os", stripped)
        self.assertNotIn("import sys", stripped)
        self.assertEqual(
            notice, "Removed redundant imports os, sys (already imported)."
        )
        self.assertEqual(jl.imported_modules, {"os", "sys"})

    def test_jupyter_strip_boilerplate_does_not_strip_unreported_import(self):
        """A module not reported by the kernel state is never stripped (regression: `import time` was stripped before it ever ran)."""
        jl = object.__new__(JupyterLanguage)  # skip kernel startup
        jl.imported_modules = {"os"}
        code = "import time\nprint(time.strftime('%H:%M:%S'))"
        stripped, notice = jl.strip_boilerplate(code)
        self.assertEqual(stripped, code)
        self.assertIsNone(notice)
        self.assertEqual(jl.imported_modules, {"os"})

    def test_jupyter_strip_boilerplate_does_not_learn_from_unstripped_code(self):
        """strip_boilerplate must not optimistically record imports from the code it strips — only the kernel's REPL-state line is authoritative (regression: a second call in the same run then stripped the import before it executed)."""
        jl = object.__new__(JupyterLanguage)  # skip kernel startup
        jl.imported_modules = {"os"}
        code = "import random\nprint(random.randint(1, 5))"
        first, _ = jl.strip_boilerplate(code)
        second, second_notice = jl.strip_boilerplate(first)
        self.assertEqual(second, code)  # still not stripped on the second (preprocess) call
        self.assertIsNone(second_notice)
        self.assertEqual(jl.imported_modules, {"os"})

    def test_jupyter_strip_boilerplate_no_notice_when_nothing_stripped(self):
        """strip_boilerplate returns (code, None) when no import is redundant."""
        jl = object.__new__(JupyterLanguage)  # skip kernel startup
        jl.imported_modules = {"os"}
        stripped, notice = jl.strip_boilerplate("import numpy as np\nnp.zeros(1)")
        self.assertEqual(stripped, "import numpy as np\nnp.zeros(1)")
        self.assertIsNone(notice)

    def test_jupyter_strip_boilerplate_is_idempotent(self):
        """Running strip_boilerplate twice yields the same result both times."""
        jl = object.__new__(JupyterLanguage)  # skip kernel startup
        jl.imported_modules = {"os"}
        first, _ = jl.strip_boilerplate("import os\nos.getcwd()")
        second, second_notice = jl.strip_boilerplate(first)
        self.assertEqual(second, first)
        self.assertIsNone(second_notice)

    def test_bash_redundant_cd_stripped(self):
        """A standalone `cd` to the tracked working directory is removed."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(
            bash._strip_redundant_cd("cd /home/user/project\nls"), "ls"
        )

    def test_bash_cd_pwd_and_dot_not_stripped(self):
        """`cd .` is a no-op spelling LLMs don't emit — it's left alone; `cd $PWD` is stripped."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(bash._strip_redundant_cd("cd .\nls"), "cd .\nls")
        self.assertEqual(bash._strip_redundant_cd("cd $PWD\nls"), "ls")
        self.assertEqual(bash._strip_redundant_cd('cd "$PWD"\nls'), "ls")

    def test_bash_redundant_cd_in_compound_chain_stripped(self):
        """A redundant `cd X` chained with `&&`, `;` or `&` is stripped, keeping the rest of the line."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(
            bash._strip_redundant_cd("cd /home/user/project && ls"), "ls"
        )
        self.assertEqual(
            bash._strip_redundant_cd("cd /home/user/project; ls -la"), "ls -la"
        )
        self.assertEqual(
            bash._strip_redundant_cd(
                "cd /home/user/project & do something | something else"
            ),
            "do something | something else",
        )

    def test_bash_cd_dot_in_compound_chain_kept(self):
        """A `cd .` even when chained is left alone — we only strip what LLMs actually emit."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(
            bash._strip_redundant_cd("cd . && ls"), "cd . && ls"
        )

    def test_bash_cd_to_other_existing_directory_kept_and_tracked(self):
        """A `cd` to a different existing directory is kept and updates the tracked cwd."""
        with tempfile.TemporaryDirectory() as d:
            bash = Bash()
            bash.cwd = "/home/user/project"
            result = bash._strip_redundant_cd(f"cd {d}\npwd")
            self.assertEqual(result, f"cd {d}\npwd")
            self.assertEqual(bash.cwd, d)

    def test_bash_cd_to_nonexistent_directory_kept_and_not_tracked(self):
        """A `cd` to a directory that doesn't exist is kept and doesn't change tracking."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        result = bash._strip_redundant_cd("cd /nonexistent_oi_xyz_dir\nls")
        self.assertEqual(result, "cd /nonexistent_oi_xyz_dir\nls")
        self.assertEqual(bash.cwd, "/home/user/project")

    def test_bash_compound_and_cd_dash_kept(self):
        """`cd X || fallback` chains and `cd -` are kept and don't change tracking."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        code = "cd /home/user/project || echo fail\ncd -\nls"
        self.assertEqual(bash._strip_redundant_cd(code), code)
        self.assertEqual(bash.cwd, "/home/user/project")

    def test_bash_relative_cd_kept_and_absolute_to_same_place_stripped(self):
        """Relative `cd`s resolve against the tracked cwd; a no-op cd to the same place is stripped."""
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "sub")
            os.mkdir(sub)
            bash = Bash()
            bash.cwd = d
            self.assertEqual(bash._strip_redundant_cd("cd sub\nls"), "cd sub\nls")
            self.assertEqual(bash.cwd, sub)
            self.assertEqual(bash._strip_redundant_cd(f"cd {sub}\nls"), "ls")

    def test_bash_quoted_redundant_cd_in_chain_stripped(self):
        """A quoted redundant target (`cd "/path" && cmd`) is stripped like an unquoted one."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(
            bash._strip_redundant_cd('cd "/home/user/project" && ls'), "ls"
        )

    def test_bash_dangling_chain_stripped_and_noticed(self):
        """`cd X &&` with nothing after (a syntax error) is dropped and still counted in the notice."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(bash._strip_redundant_cd("cd /home/user/project &&"), "")
        self.assertEqual(
            bash._pending_notice,
            "Removed redundant cd /home/user/project (already in that directory).",
        )

    def test_bash_dangling_chain_consumed_by_next_line(self):
        """`cd X &&` followed by a real command on the next line keeps that command."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        result = bash._strip_redundant_cd("cd /home/user/project &&\nls")
        self.assertEqual(result, "ls")

    def test_bash_trailing_semicolon_cd_stripped(self):
        """`cd X;` (semicolon, nothing after) is dropped like the standalone form."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(bash._strip_redundant_cd("cd /home/user/project;"), "")

    def test_bash_no_space_chain_operators_stripped(self):
        """Chain operators without surrounding spaces (`cd X&&ls`, `cd X;ls`, `cd X&ls`) are handled."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(
            bash._strip_redundant_cd("cd /home/user/project&&ls"), "ls"
        )
        self.assertEqual(
            bash._strip_redundant_cd("cd /home/user/project;ls"), "ls"
        )
        self.assertEqual(
            bash._strip_redundant_cd("cd /home/user/project&ls"), "ls"
        )

    def test_bash_bare_cd_kept(self):
        """A bare `cd` (no target) goes to $HOME — kept, since we can't verify it's redundant."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(bash._strip_redundant_cd("cd\nls"), "cd\nls")

    def test_bash_cd_with_pipe_or_redirect_kept(self):
        """A `cd` piped or redirected is not a directory-change command and is kept whole."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(
            bash._strip_redundant_cd("cd /home/user/project | wc"),
            "cd /home/user/project | wc",
        )
        self.assertEqual(
            bash._strip_redundant_cd("cd /home/user/project 2>/dev/null"),
            "cd /home/user/project 2>/dev/null",
        )

    def test_bash_cd_unclosed_quote_kept(self):
        """An unterminated quoted target can't be parsed — the line is kept untouched."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(
            bash._strip_redundant_cd("cd '/home/user/project"), "cd '/home/user/project"
        )

    def test_bash_double_redundant_cd_both_stripped(self):
        """A standalone and a chained redundant cd to the same dir are both removed."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        result = bash._strip_redundant_cd(
            "cd /home/user/project\ncd /home/user/project && ls"
        )
        self.assertEqual(result, "ls")

    def test_bash_redundant_cd_before_real_cd_keeps_real_cd(self):
        """Only the redundant cd is dropped; a later real `cd` is kept and tracked."""
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "sub")
            os.mkdir(sub)
            bash = Bash()
            bash.cwd = d
            result = bash._strip_redundant_cd(f"cd {d} && cd {sub} && ls")
            self.assertEqual(result, f"cd {sub} && ls")
            self.assertEqual(bash.cwd, sub)

    def test_bash_cd_target_with_trailing_slash_stripped(self):
        """A redundant target spelled with a trailing slash still resolves to the cwd and is stripped."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(
            bash._strip_redundant_cd("cd /home/user/project/ && ls"), "ls"
        )

    def test_bash_cd_env_var_target_kept(self):
        """An env-var target (`cd $HOME`) is not mistaken for the cwd when it differs."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        self.assertEqual(
            bash._strip_redundant_cd("cd $HOME && ls"), "cd $HOME && ls"
        )

    def test_powershell_redundant_cd_stripped_with_aliases_and_case_insensitivity(self):
        """PowerShell strips `cd`/`Set-Location`/`sl` to the current dir, case-insensitively, even when chained."""
        ps = _StubCwdShell(cd_commands=("cd", "Set-Location", "sl"), cd_ignore_case=True)
        ps.cwd = "/home/user/project"
        self.assertEqual(
            ps._strip_redundant_cd("cd /home/user/project; Get-ChildItem"),
            "Get-ChildItem",
        )
        self.assertEqual(
            ps._strip_redundant_cd("Set-Location /home/user/project && dir"),
            "dir",
        )
        self.assertEqual(
            ps._strip_redundant_cd("SL /home/user/project & dir"), "dir"
        )
        self.assertEqual(
            ps._strip_redundant_cd("CD /home/user/project; dir"), "dir"
        )

    def test_powershell_redundant_cd_to_other_dir_kept(self):
        """A PowerShell cd to a different existing directory is kept."""
        with tempfile.TemporaryDirectory() as d:
            ps = _StubCwdShell(cd_commands=("cd", "Set-Location", "sl"), cd_ignore_case=True)
            ps.cwd = "/home/user/project"
            result = ps._strip_redundant_cd(f"cd {d}; dir")
            self.assertEqual(result, f"cd {d}; dir")

    def test_cmd_redundant_cd_stripped_with_drive_flag(self):
        """cmd strips `cd /d` and plain `cd` to the current dir even when chained with `&`/`&&`."""
        cmd = _StubCwdShell(cd_option_prefixes=("/d",), cd_chain_operators=("&&", "&"))
        cmd.cwd = "/home/user/project"
        self.assertEqual(
            cmd._strip_redundant_cd("cd /d /home/user/project & dir"), "dir"
        )
        self.assertEqual(
            cmd._strip_redundant_cd("cd /home/user/project && dir"), "dir"
        )

    def test_cmd_cd_percent_cd_and_semicolon_handling(self):
        """cmd treats `%CD%` as the current dir; `;` is not a cmd separator so `cd X; ls` is kept whole."""
        cmd = _StubCwdShell(cd_option_prefixes=("/d",), cd_chain_operators=("&&", "&"))
        cmd.cwd = "/home/user/project"
        self.assertEqual(cmd._strip_redundant_cd("cd %CD% && dir"), "dir")
        self.assertEqual(
            cmd._strip_redundant_cd("cd /home/user/project; dir"),
            "cd /home/user/project; dir",
        )

    def test_cwd_marker_filter_works_on_all_shell_configs(self):
        """The `##oi_pwd##` marker is filtered and updates cwd for any shell config."""
        ps = _StubCwdShell(cd_commands=("cd", "Set-Location", "sl"), cd_ignore_case=True)
        self.assertIsNone(ps.line_postprocessor("##oi_pwd##/new/dir"))
        self.assertEqual(ps.cwd, "/new/dir")

        cmd = _StubCwdShell(cd_option_prefixes=("/d",), cd_chain_operators=("&&", "&"))
        self.assertIsNone(cmd.line_postprocessor("##oi_pwd##C:\\new\\dir"))
        self.assertEqual(cmd.cwd, "C:\\new\\dir")

    def test_bash_sets_cd_removal_notice(self):
        """Removing a redundant cd sets _pending_notice naming the target directory."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        result = bash._strip_redundant_cd("cd /home/user/project && ls")
        self.assertEqual(result, "ls")
        self.assertEqual(
            bash._pending_notice,
            "Removed redundant cd /home/user/project (already in that directory).",
        )

    def test_bash_no_cd_notice_when_nothing_removed(self):
        """_pending_notice stays None when no cd prefix is stripped."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        bash._pending_notice = None
        bash._strip_redundant_cd("ls")
        self.assertIsNone(bash._pending_notice)

    def test_cmd_sets_cd_removal_notice_with_drive_flag(self):
        """cmd records the stripped cd target in _pending_notice even with the /d form."""
        cmd = _StubCwdShell(cd_option_prefixes=("/d",), cd_chain_operators=("&&", "&"))
        cmd.cwd = "/home/user/project"
        result = cmd._strip_redundant_cd("cd /d /home/user/project & dir")
        self.assertEqual(result, "dir")
        self.assertEqual(
            cmd._pending_notice,
            "Removed redundant cd /home/user/project (already in that directory).",
        )

    def test_bash_strip_boilerplate_returns_stripped_and_notice(self):
        """strip_boilerplate returns the code minus the redundant cd plus the notice."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        stripped, notice = bash.strip_boilerplate("cd /home/user/project && ls")
        self.assertEqual(stripped, "ls")
        self.assertEqual(
            notice,
            "Removed redundant cd /home/user/project (already in that directory).",
        )
        # Idempotent: a second pass strips nothing more.
        again, second_notice = bash.strip_boilerplate(stripped)
        self.assertEqual(again, "ls")
        self.assertIsNone(second_notice)

    def test_bash_strip_boilerplate_no_notice_when_nothing_removed(self):
        """strip_boilerplate returns (code, None) when there is no redundant cd."""
        bash = Bash()
        bash.cwd = "/home/user/project"
        stripped, notice = bash.strip_boilerplate("ls")
        self.assertEqual(stripped, "ls")
        self.assertIsNone(notice)

    def test_bash_peek_strip_does_not_advance_cwd_or_strip_real_cd(self):
        """The respond-path peek must not pre-apply a kept cd (regression: `cd /home/user && pwd` printed the old dir because the peek advanced cwd and the run then stripped it)."""
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "sub")
            os.mkdir(sub)
            bash = Bash()
            bash.cwd = d

            # Peek (respond): a cd to a different existing dir is kept and does
            # NOT advance the tracked cwd — it hasn't actually run yet.
            stripped, notice = bash.strip_boilerplate(f"cd {sub} && ls")
            self.assertEqual(stripped, f"cd {sub} && ls")
            self.assertIsNone(notice)
            self.assertEqual(bash.cwd, d)

            # Actual run (preprocess): the same cd is still not redundant
            # (tracked cwd unchanged), so it is kept and the cwd advances.
            result = bash._strip_redundant_cd(stripped, track=True)
            self.assertEqual(result, f"cd {sub} && ls")
            self.assertEqual(bash.cwd, sub)

    def test_code_block_sync_stored_code_updates_displayed_code_once(self):
        """sync_stored_code swaps in the stripped code so the finalized panel prints it only once."""
        from interpreter.terminal_interface.components.code_block import CodeBlock

        block = object.__new__(CodeBlock)  # skip Rich/terminal setup
        block.code = "import os\nprint(1)"
        self.assertTrue(block.sync_stored_code("print(1)"))
        self.assertEqual(block.code, "print(1)")
        # Idempotent: once synced, a repeat call is a no-op (prevents re-printing).
        self.assertFalse(block.sync_stored_code("print(1)"))
        self.assertEqual(block.code, "print(1)")

    def test_code_block_sync_stored_code_clears_on_empty(self):
        """sync_stored_code clears the displayed code when the stored version is empty — a block fully stripped (e.g. a lone redundant cd) must not leave stale code on screen (regression)."""
        from interpreter.terminal_interface.components.code_block import CodeBlock

        block = object.__new__(CodeBlock)
        block.code = "cd /home/user/project"
        self.assertTrue(block.sync_stored_code(""))
        self.assertEqual(block.code, "")

    def test_code_block_sync_stored_code_ignores_identical(self):
        """sync_stored_code leaves the block untouched when stored code is identical (idempotent — prevents re-printing)."""
        from interpreter.terminal_interface.components.code_block import CodeBlock

        block = object.__new__(CodeBlock)
        block.code = "x = 1"
        self.assertFalse(block.sync_stored_code("x = 1"))
        self.assertEqual(block.code, "x = 1")

    def test_bash_line_postprocessor_filters_pwd_marker(self):
        """The `##oi_pwd##` marker is filtered from output and updates the tracked cwd."""
        bash = Bash()
        self.assertIsNone(bash.line_postprocessor("##oi_pwd##/new/dir"))
        self.assertEqual(bash.cwd, "/new/dir")

    def test_bash_preprocess_appends_pwd_marker_before_end_marker(self):
        """preprocess_code appends the `##oi_pwd##$PWD` echo just before the end-of-execution marker."""
        bash = Bash()
        out = bash.preprocess_code("echo hi")
        self.assertLess(
            out.index('echo "##oi_pwd##$PWD"'),
            out.index('echo "##end_of_execution##"'),
        )


class _FakeTerminal:
    """Duck-types terminal.get_language/get_language_instance for respond() notice tests."""

    def __init__(self, instances=None):
        self.instances = instances or {}
        # Every language in the message's format must be "supported".
        self._known = {
            "python",
            "bash",
            "powershell",
            "cmd",
            "javascript",
            "r",
            "ruby",
            "applescript",
            "java",
            "perl",
        }

    def get_language(self, language):
        return language if language in self._known else None

    def get_language_instance(self, language):
        return self.instances.get(language)


class TestRespondNotices(unittest.TestCase):
    """respond() reports silent code rewrites via the confirmation chunk's `removed` notice."""

    def _confirmation(self, messages, instances=None):
        from interpreter.core.respond import respond

        class FakeInterpreter:
            verbose = False
            toolbox = type(
                "Toolbox",
                (),
                {"import_toolbox_api": True},
            )()

        interp = FakeInterpreter()
        interp.messages = messages
        interp.terminal = _FakeTerminal(instances)
        with patch(
            "interpreter.core.respond.assemble_system_message", return_value=""
        ):
            for chunk in respond(interp):
                if chunk.get("type") == "confirmation":
                    return chunk
        self.fail("respond() never yielded a confirmation chunk")

    def test_functions_execute_wrapper_extraction_gets_notice(self):
        """A `functions.execute(...)` wrapper is unwrapped into plain code and reported via `removed`."""
        chunk = self._confirmation(
            [
                {"role": "user", "type": "message", "content": "hi"},
                {
                    "role": "assistant",
                    "type": "code",
                    "format": "python",
                    "content": 'functions.execute({"language": "python", "code": "print(1)"})',
                },
            ]
        )
        content = chunk["content"]
        self.assertEqual(content["content"], "print(1)")
        self.assertIn("functions.execute()", content["removed"])

    def test_executeexecute_suffix_removal_gets_notice(self):
        """A stray trailing `executeexecute` token is removed and reported via `removed`."""
        chunk = self._confirmation(
            [
                {"role": "user", "type": "message", "content": "hi"},
                {
                    "role": "assistant",
                    "type": "code",
                    "format": "bash",
                    "content": "ls executeexecute",
                },
            ]
        )
        content = chunk["content"]
        self.assertEqual(content["content"].strip(), "ls")
        self.assertIn("executeexecute", content["removed"])

    def test_json_language_block_extraction_gets_notice(self):
        """A JSON `{language: ...}` code block is unwrapped into plain code and reported via `removed`."""
        chunk = self._confirmation(
            [
                {"role": "user", "type": "message", "content": "hi"},
                {
                    "role": "assistant",
                    "type": "code",
                    "format": "bash",
                    "content": '{language: "bash", code: "ls"}',
                },
            ]
        )
        content = chunk["content"]
        self.assertEqual(content["content"], "ls")
        self.assertIn("language", content["removed"])

    def test_hardcoded_fix_and_cd_strip_notices_combined(self):
        """A hardcoded-fix notice and the redundant-cd notice are joined into a single `removed` string."""
        from interpreter.core.terminal.languages.bash import Bash

        bash = Bash()
        bash.cwd = os.getcwd()
        chunk = self._confirmation(
            [
                {"role": "user", "type": "message", "content": "hi"},
                {
                    "role": "assistant",
                    "type": "code",
                    "format": "bash",
                    "content": f'cd "{bash.cwd}" && ls executeexecute',
                },
            ],
            instances={"bash": bash},
        )
        content = chunk["content"]
        self.assertEqual(content["content"].strip(), "ls")
        self.assertIn("executeexecute", content["removed"])
        self.assertIn("already in that directory", content["removed"])

    def test_no_notice_when_nothing_rewritten(self):
        """When the code is run as written, the confirmation chunk carries no `removed` notice."""
        chunk = self._confirmation(
            [
                {"role": "user", "type": "message", "content": "hi"},
                {
                    "role": "assistant",
                    "type": "code",
                    "format": "python",
                    "content": "print('hi')",
                },
            ]
        )
        self.assertIsNone(chunk["content"]["removed"])

    def test_import_toolbox_stripped_with_notice(self):
        """`import toolbox` is stripped (toolbox is injected into the kernel) and reported — the old guard raised after approval instead (regression)."""
        chunk = self._confirmation(
            [
                {"role": "user", "type": "message", "content": "hi"},
                {
                    "role": "assistant",
                    "type": "code",
                    "format": "python",
                    "content": "import toolbox\nr = toolbox.web.fetch('http://example.com')",
                },
            ]
        )
        content = chunk["content"]
        self.assertEqual(content["format"], "python")
        self.assertNotIn("import toolbox", content["content"])
        self.assertIn("r = toolbox.web.fetch('http://example.com')", content["content"])
        self.assertIn("import toolbox", content["removed"])

    def test_import_toolbox_with_comment_stripped(self):
        """An `import toolbox  # comment` line is stripped like the plain form."""
        chunk = self._confirmation(
            [
                {"role": "user", "type": "message", "content": "hi"},
                {
                    "role": "assistant",
                    "type": "code",
                    "format": "python",
                    "content": "import toolbox  # needed\nprint(toolbox)",
                },
            ]
        )
        content = chunk["content"]
        self.assertNotIn("import toolbox", content["content"])
        self.assertIn("print(toolbox)", content["content"])

    def test_import_toolbox_as_kept_for_guard(self):
        """Aliased `import toolbox as tb` is NOT stripped by the line filter — the post-confirmation guard still handles it."""
        chunk = self._confirmation(
            [
                {"role": "user", "type": "message", "content": "hi"},
                {
                    "role": "assistant",
                    "type": "code",
                    "format": "python",
                    "content": "import toolbox as tb\nprint(tb)",
                },
            ]
        )
        self.assertIn("import toolbox as tb", chunk["content"]["content"])

    def test_import_toolbox_not_stripped_when_api_disabled(self):
        """When the toolbox API isn't injected, `import toolbox` is left alone."""
        from interpreter.core.respond import respond

        class FakeInterpreter:
            verbose = False
            toolbox = type(
                "Toolbox",
                (),
                {"import_toolbox_api": False},
            )()

        interp = FakeInterpreter()
        interp.messages = [
            {"role": "user", "type": "message", "content": "hi"},
            {
                "role": "assistant",
                "type": "code",
                "format": "python",
                "content": "import toolbox\nprint(toolbox)",
            },
        ]
        interp.terminal = _FakeTerminal()
        with patch(
            "interpreter.core.respond.assemble_system_message", return_value=""
        ):
            for chunk in respond(interp):
                if chunk.get("type") == "confirmation":
                    self.assertIn("import toolbox", chunk["content"]["content"])
                    return
        self.fail("respond() never yielded a confirmation chunk")


class TestTerminalInterfaceNotices(unittest.TestCase):
    """terminal_interface handles removed-notice chunks without crashing."""

    def _drive(self, chunks, auto_run_mode="all"):
        from interpreter.terminal_interface.terminal_interface import (
            terminal_interface,
        )

        class FakeLLM:
            supports_vision = False
            vision_renderer = None

        class FakeInterpreter:
            verbose = False
            offline = False
            os = None
            plain_text_display = True
            highlight_active_line = False

            def __init__(self, chunks, auto_run_mode):
                self.auto_run_mode = auto_run_mode
                self.llm = FakeLLM()
                self.messages = [{"role": "user", "type": "message", "content": "hi"}]

            def chat(self, message, display=False, stream=True):
                yield from chunks

            def display_message(self, text):
                pass

        interp = FakeInterpreter(chunks, auto_run_mode)
        # A non-empty message makes the loop run exactly one pass (not interactive).
        return list(terminal_interface(interp, "test message"))

    def test_non_confirmation_chunk_does_not_crash(self):
        """A message chunk after no confirmation must not raise UnboundLocalError (regression: the removed-notice print was placed outside the confirmation branch and referenced the unbound variable)."""
        chunks = [{"role": "assistant", "type": "message", "content": "hello"}]
        out = self._drive(chunks)
        self.assertEqual(len(out), 1)

    def test_auto_run_notice_printed(self):
        """In auto-run mode a stripped confirmation prints the notice (no prompt appears)."""
        from io import StringIO
        from unittest.mock import patch

        chunk = {
            "role": "computer",
            "type": "confirmation",
            "format": "execution",
            "content": {
                "type": "code",
                "format": "bash",
                "content": "ls",
                "removed": "Removed redundant cd /x (already in that directory).",
            },
        }
        with patch("sys.stdout", new_callable=StringIO) as fake_out:
            self._drive([chunk])
            self.assertIn("Removed redundant cd /x", fake_out.getvalue())

    def test_auto_run_followed_by_message_chunk_does_not_crash(self):
        """Auto-run confirmation followed by a plain message chunk must not crash (the removed_notice variable is only read inside the confirmation handler)."""
        chunks = [
            {
                "role": "computer",
                "type": "confirmation",
                "format": "execution",
                "content": {
                    "type": "code",
                    "format": "bash",
                    "content": "ls",
                    "removed": "Removed redundant cd /x (already in that directory).",
                },
            },
            {"role": "assistant", "type": "message", "content": "done"},
        ]
        out = self._drive(chunks)
        self.assertEqual(len(out), 2)


class TestCwdParsingFuzz(unittest.TestCase):
    """Property-based (hypothesis) tests for the cd-target parser and stripper.

    These fuzz the tokenizer with arbitrary inputs to catch parsing edge
    cases a hand-written test table would miss — quoting/escaping
    interactions, chain operators adjacent to the target, and the split of a
    target word at every possible character. NOTE: hypothesis is a dev-branch
    experiment; we're not sure we'll keep it as a dependency when these
    features are merged into main, so don't build other tests on it.
    """

    # hypothesis's internal constant-collection scans loaded modules for
    # constants and tries to import `pynput`, which raises on headless Linux
    # (no X display) — unrelated to our parser. Block it so the fuzzer can run.
    try:
        import sys as _sys

        if "pynput" in _sys.modules:
            del _sys.modules["pynput"]
    except Exception:
        pass

    def setUp(self):
        # Bash is the one config with backslash-unescaping enabled.
        self.bash = _StubCwdShell(cd_unescape_backslashes=True)
        self.bash.cwd = "/fuzz/start"

    # A target word: printable chars, no whitespace, and none of the chars
    # that terminate/split the target (backslash, ; | &, quotes). Escaped
    # variants (with backslashes) are generated separately below.
    safe_word = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/-"
    )

    def test_parse_cd_never_raises(self):
        """_parse_cd never raises and returns a 3-tuple for arbitrary input (including non-cd lines)."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        @given(st.text(max_size=80))
        @settings(max_examples=300)
        def run(line):
            result = self.bash._parse_cd(line)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)

        run()

    def test_parse_cd_simple_word_target_is_exact(self):
        """A plain target word parses exactly, with no chain suffix."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        @given(st.text(alphabet=self.safe_word, min_size=1, max_size=40))
        @settings(max_examples=300)
        def run(word):
            target, after, ok = self.bash._parse_cd(f"cd {word}")
            self.assertTrue(ok)
            self.assertEqual(target, word)
            self.assertEqual(after, "")

        run()

    def test_parse_cd_word_adjacent_to_chain_operator(self):
        """A chain operator directly adjacent to the target (`cd X;ls`, `cd X&&ls`) splits it off, not into the target."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        operators = st.sampled_from(["&&", "||", ";", "&"])

        @given(
            st.text(alphabet=self.safe_word, min_size=1, max_size=20),
            operators,
            st.text(alphabet="ls", min_size=1, max_size=10),
        )
        @settings(max_examples=300)
        def run(word, op, cmd):
            target, after, ok = self.bash._parse_cd(f"cd {word}{op}{cmd}")
            self.assertTrue(ok)
            self.assertEqual(target, word)
            self.assertTrue(after.startswith(op))

        run()

    def test_parse_cd_word_after_whitespace_chain(self):
        """A chain operator after whitespace (`cd X && ls`) splits with the operator at the head of `after`."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        operators = st.sampled_from(["&&", "||", ";", "&"])

        @given(
            st.text(alphabet=self.safe_word, min_size=1, max_size=20),
            operators,
            st.text(alphabet="ls", min_size=1, max_size=10),
        )
        @settings(max_examples=300)
        def run(word, op, cmd):
            target, after, ok = self.bash._parse_cd(f"cd {word} {op} {cmd}")
            self.assertTrue(ok)
            self.assertEqual(target, word)
            self.assertTrue(after.startswith(op))

        run()

    def test_parse_cd_single_pipe_or_redirect_unparseable(self):
        """A bare `|` or redirect after the target makes the line unparseable (a cd piped/redirected isn't a directory change)."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        suffixes = st.sampled_from(["| wc", "> /dev/null", "2>/dev/null"])

        @given(
            st.text(alphabet=self.safe_word, min_size=1, max_size=20),
            suffixes,
        )
        @settings(max_examples=200)
        def run(word, suffix):
            target, after, ok = self.bash._parse_cd(f"cd {word} {suffix}")
            self.assertFalse(ok)

        run()

    def test_parse_cd_quoted_target_preserves_content(self):
        """A quoted target parses to exactly the quoted content, preserving spaces and punctuation (an empty `""` is a valid no-op target)."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        # Quotable content with NO embedded double-quote (which the naive
        # first-quote parser can't represent faithfully). Spaces and shell
        # punctuation are the interesting cases.
        quoted_safe = self.safe_word + " " + "~`!@#$%^&*()+-=[]{}<>,?'"

        @given(st.text(alphabet=quoted_safe, min_size=0, max_size=30))
        @settings(max_examples=300)
        def run(content):
            target, after, ok = self.bash._parse_cd(f'cd "{content}"')
            # Even empty (`cd ""`) parses with an empty target.
            self.assertTrue(ok)
            self.assertEqual(target, content)
            self.assertEqual(after, "")

        run()

    def test_parse_cd_escaped_space_stays_in_target(self):
        """A backslash-escaped space (`My\\ Documents`) stays inside the target word."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        @given(
            st.text(alphabet=self.safe_word, min_size=1, max_size=15),
            st.text(alphabet=self.safe_word, min_size=1, max_size=15),
        )
        @settings(max_examples=200)
        def run(part1, part2):
            target, after, ok = self.bash._parse_cd(f"cd {part1}\\ {part2}")
            self.assertTrue(ok)
            self.assertEqual(target, f"{part1} {part2}")
            self.assertEqual(after, "")

        run()

    def test_bash_strips_redundant_cd_with_escaped_space(self):
        """A redundant cd whose path contains a backslash-escaped space is stripped with a notice (the bug that motivated this work)."""
        b = Bash()
        b.cwd = "/home/user/documents/My Documents"
        stripped = b._strip_redundant_cd(
            "cd /home/user/documents/My\\ Documents/\npdftotext report.pdf 2>/dev/null | grep -c PATTERN"
        )
        self.assertEqual(stripped, "pdftotext report.pdf 2>/dev/null | grep -c PATTERN")
        self.assertEqual(
            b._pending_notice,
            "Removed redundant cd /home/user/documents/My Documents/ (already in that directory).",
        )
        # The peek must not advance cwd.
        self.assertEqual(b.cwd, "/home/user/documents/My Documents")

    def test_bash_keeps_cd_with_escaped_space_to_other_dir(self):
        """A cd to a DIFFERENT dir whose path has an escaped space is kept and tracked correctly."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "sub dir")
            os.mkdir(sub)
            esc = sub.replace(" ", "\\ ")
            b = Bash()
            b.cwd = d
            result = b._strip_redundant_cd(f"cd {esc} && ls")
            self.assertEqual(result, f"cd {esc} && ls")
            self.assertEqual(b.cwd, sub)

    def test_parse_cd_escaped_char_round_trips(self):
        """Any `\\X` escape unescapes to `X`, so the target compares equal to the literal path."""
        from hypothesis import given, settings, HealthCheck
        from hypothesis import strategies as st

        @given(
            st.lists(
                st.tuples(st.sampled_from(self.safe_word), st.sampled_from(self.safe_word)),
                min_size=1,
                max_size=10,
            )
        )
        @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def run(pairs):
            escaped = "".join(f"\\{a}{b}" for a, b in pairs)
            target, after, ok = self.bash._parse_cd(f"cd {escaped}")
            self.assertTrue(ok)
            self.assertEqual(target, escaped.replace("\\", ""))
            self.assertEqual(after, "")

        run()

    def test_peek_strip_idempotent_on_arbitrary_code(self):
        """The respond peek (track=False) is idempotent for any multi-line code."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        line = st.text(alphabet=self.safe_word + " ;&|\\", max_size=40)

        @given(st.lists(line, min_size=0, max_size=10))
        @settings(max_examples=300)
        def run(lines):
            code = "\n".join(lines)
            first = self.bash._strip_redundant_cd(code, track=False)
            second = self.bash._strip_redundant_cd(first, track=False)
            self.assertEqual(second, first)
            # The peek never mutates cwd.
            self.assertEqual(self.bash.cwd, "/fuzz/start")

        run()

    def test_strip_never_changes_cwd_when_tracking_disabled(self):
        """track=False (the respond peek) never advances the tracked cwd, for any code."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        line = st.text(alphabet=self.safe_word + " ;&|\\", max_size=40)

        @given(st.lists(line, min_size=0, max_size=10))
        @settings(max_examples=200)
        def run(lines):
            code = "\n".join(lines)
            before = self.bash.cwd
            self.bash._strip_redundant_cd(code, track=False)
            self.assertEqual(self.bash.cwd, before)

        run()


if __name__ == "__main__":
    unittest.main()
