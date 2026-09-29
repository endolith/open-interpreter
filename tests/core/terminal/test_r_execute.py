import shutil
import unittest

from interpreter.core.terminal.languages.r import R


@unittest.skipUnless(shutil.which("R"), "R not installed")
class TestRReplOutput(unittest.TestCase):
    """Drive the R backend end to end over a real R process.

    R echoes every line of input back on stdout, including when stdin is a
    plain pipe, and those echoes interleave with real output. The backend used
    to suppress them by skipping a computed number of lines, which cannot work
    when the two are interleaved: the skip consumed real output, and an echo
    that slipped through carried the end-of-execution marker, ending the block
    early and printing the echoed source ("> cat(...)") as if it were output.
    These tests pin the corrected behaviour, which relies on --no-echo instead.
    """

    def setUp(self):
        self.r = R()

    def tearDown(self):
        self.r.terminate()

    def _run(self, code):
        """Run a block and return the non-active-line console output."""
        return [
            str(chunk["content"])
            for chunk in self.r.run(code)
            if chunk.get("format") != "active_line" and chunk.get("content")
        ]

    def test_start_cmd_suppresses_echo(self):
        """R is started with --no-echo so the output stream has no echoed source.

        Without it R interleaves echoed input with real output, which is the
        condition this backend exists to work around.
        """
        self.assertIn("--no-echo", self.r.start_cmd)

    def test_no_line_count_suppression(self):
        """The backend must not skip lines by count.

        Echoes interleave with real output, so a count is always wrong: too
        large swallows genuine output (and the end-of-execution marker, which
        hangs the block forever waiting for a done signal that never arrives),
        too small leaks an echo that ends the block early.
        """
        self.assertFalse(
            hasattr(R(), "code_line_count"),
            "count-based echo suppression cannot work against interleaved output",
        )

    def test_simple_print_returns_only_the_value(self):
        """A basic print yields the value, with no echoed source and no stray blanks."""
        output = self._run('print("hello world")')
        self.assertEqual(output, ['"hello world"'])

    def test_error_reports_message_without_protocol_marker(self):
        """An R error surfaces its message, not the internal marker text."""
        output = self._run("print(nosuchvar)")
        joined = "".join(output)
        self.assertIn("nosuchvar", joined)
        self.assertNotIn("##execution_error##", joined)
        self.assertNotIn("##end_of_execution##", joined)

    def test_state_persists_across_blocks(self):
        """The REPL stays live: a variable set in one block is visible in the next."""
        self._run("x <- 41")
        self.assertEqual(self._run("print(x + 1)"), ["42"])

    def test_multi_line_block_produces_no_stray_output(self):
        """Several lines in one block do not leak markers or blank lines."""
        output = self._run("a <- 1\nb <- 2\nprint(a + b)")
        self.assertEqual(output, ["3"])

    def test_no_echoed_source_in_output(self):
        """No output line may contain a fragment of the wrapper we injected."""
        output = self._run('print("check")')
        joined = "".join(output)
        for fragment in ("tryCatch", "cat(", "##", 'print("check")'):
            self.assertNotIn(fragment, joined)


if __name__ == "__main__":
    unittest.main()
