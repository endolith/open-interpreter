"""Code panels must render code verbatim, without trimming blank lines.

Regression: CodeBlock called ``code.strip().split("\\n")`` before rendering, so
leading/trailing blank lines were trimmed from every code panel. An append or
edit block that starts with blank lines then looked different in the panel from
what actually runs.
"""

import unittest

from rich.console import Console

from interpreter.terminal_interface.components.code_block import CodeBlock


class _RecordingLive:
    """Stand-in for ``rich.live.Live`` that prints the updated group."""

    def __init__(self, console):
        self.console = console

    def update(self, group):
        self.console.print(group)

    def refresh(self):
        pass

    def stop(self):
        pass


def render_code(content, language="python", cursor=False):
    """Render content through ``CodeBlock.refresh`` and return console text."""
    block = CodeBlock(interpreter=None)
    block.language = language
    block.code = content
    block.output = ""
    console = Console(width=100, record=True)
    block.live = _RecordingLive(console)
    with console.capture() as capture:
        block.refresh(cursor=cursor)
    return capture.get()


class TestCodeBlockWhitespace(unittest.TestCase):
    def test_leading_blank_lines_are_shown(self):
        """Leading blank lines must survive into the rendered panel."""
        plain = render_code("charlie")
        padded = render_code("\n\ncharlie")

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
        self.assertNotEqual(render_code("\n\n"), "")
        self.assertNotEqual(render_code("\n\n"), render_code("\n"))
