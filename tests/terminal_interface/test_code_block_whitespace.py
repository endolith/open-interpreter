"""Code panels must render code verbatim, without trimming blank lines.

Regression: CodeBlock stripped leading/trailing whitespace from every code
block it displayed, so an append block starting with blank lines (which the
dry-run diff shows) rendered in the fence without them. The panel must show
exactly what will run.
"""

import unittest

from rich.console import Console

from interpreter.terminal_interface.components.code_block import CodeBlock


def render_code(content, language="diff"):
    """Render content through the finalized code panel; return the text."""
    block = CodeBlock(interpreter=None)
    block.language = language
    console = Console(width=100, record=True)
    block.live.console = console
    with console.capture() as capture:
        block._print_permanent_block(content, "code")
    return capture.get()


class TestCodeBlockWhitespace(unittest.TestCase):
    def test_leading_blank_lines_are_shown(self):
        """Leading blank lines must survive into the rendered panel."""
        plain = render_code("charlie")
        padded = render_code("\n\ncharlie\n")
        self.assertIn("charlie", plain)
        self.assertIn("charlie", padded)
        self.assertNotEqual(
            plain,
            padded,
            "leading blank lines were trimmed from the code panel",
        )

    def test_trailing_newline_renders(self):
        """A trailing newline must render instead of being trimmed away."""
        without = render_code("charlie")
        with_newline = render_code("charlie\n")
        self.assertNotEqual(
            without,
            with_newline,
            "trailing newline was trimmed from the code panel",
        )

    def test_whitespace_only_code_still_renders(self):
        """A whitespace-only block must not vanish entirely."""
        out = render_code("\n\n")
        self.assertNotEqual(out, "")


if __name__ == "__main__":
    unittest.main()
