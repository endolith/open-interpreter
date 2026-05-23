import shutil
import subprocess
import unittest
from pathlib import Path

# Project root: tests/core/terminal -> parents[3]
REPL_PATH = (
    Path(__file__).resolve().parents[3]
    / "interpreter"
    / "core"
    / "terminal"
    / "languages"
    / "perl_repl.pl"
)


@unittest.skipUnless(shutil.which("perl"), "perl not installed")
class TestPerlReplScript(unittest.TestCase):
    """Drive perl_repl.pl directly (same protocol as execute perl)."""

    def _communicate_block(self, code, extra_marker_bytes=b""):
        proc = subprocess.Popen(
            ["perl", str(REPL_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        body = code.rstrip() + "\n"
        marker = b"__OI_END__\n" + extra_marker_bytes
        proc.stdin.write(body.encode("utf-8") + marker)
        proc.stdin.flush()
        out, err = proc.communicate(timeout=10)
        return out.decode("utf-8", errors="replace"), err, proc.returncode

    def test_block_runs_and_emits_end_marker(self):
        out, err, rc = self._communicate_block('print "hello\n";')
        self.assertEqual(rc, 0)
        self.assertIn("hello", out)
        self.assertIn("##end_of_execution##", out)
        self.assertEqual(err, b"")

    def test_marker_with_cr_suffix_still_completes(self):
        """chomp on __OI_END__ must accept CRLF markers (Windows text=True bug)."""
        out, _, rc = self._communicate_block("print \"ok\n\";", extra_marker_bytes=b"\r")
        self.assertEqual(rc, 0)
        self.assertIn("ok", out)
        self.assertIn("##end_of_execution##", out)

    def test_bare_variable_persists_across_blocks(self):
        proc = subprocess.Popen(
            ["perl", str(REPL_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        proc.stdin.write(b'$counter = 41;\n__OI_END__\n')
        proc.stdin.write(b'$counter++;\nprint "$counter\n";\n__OI_END__\n')
        proc.stdin.flush()
        out, err = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(err, b"")
        text = out.decode("utf-8", errors="replace")
        self.assertIn("42", text)
        self.assertEqual(text.count("##end_of_execution##"), 2)
        proc.stdin.close()

    def test_my_variable_does_not_persist(self):
        proc = subprocess.Popen(
            ["perl", str(REPL_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        proc.stdin.write(b'my $hidden = "secret";\n__OI_END__\n')
        proc.stdin.write(b'print "$hidden\n";\n__OI_END__\n')
        proc.stdin.flush()
        out, _ = proc.communicate(timeout=10)
        text = out.decode("utf-8", errors="replace")
        self.assertNotIn("secret", text)
        proc.stdin.close()


@unittest.skipUnless(shutil.which("perl"), "perl not installed")
class TestPerlLanguageClass(unittest.TestCase):
    def test_preprocess_appends_block_marker(self):
        from interpreter.core.terminal.languages.perl import Perl

        p = Perl()
        self.assertTrue(p.binary_stdio)
        self.assertTrue(p.preprocess_code("print 1;\n").endswith("__OI_END__\n"))

    def test_run_yields_output_and_finishes(self):
        from interpreter.core.terminal.languages.perl import Perl

        p = Perl()
        chunks = []
        for chunk in p.run('print "from_class\n";'):
            if chunk.get("format") == "output":
                chunks.append(chunk["content"])
        p.terminate()
        combined = "".join(chunks)
        self.assertIn("from_class", combined)


if __name__ == "__main__":
    unittest.main()
