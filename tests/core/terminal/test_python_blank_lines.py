import ast
import os
import unittest
from unittest.mock import patch

from interpreter.core.terminal.languages.jupyter_language import preprocess_python


def _string_values(code):
    """Every constant assigned at top level in `code`, in order.

    Reads the *value* out of the AST rather than comparing source text, because
    with active-line detection on, `ast.unparse` rewrites a triple-quoted
    literal as an escaped single-line repr. The text differs while the value is
    identical, and the value is what a user's file body depends on.
    """
    values = []
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            values.append(node.value.value)
    return values


class TestBlankLinesInsideStringsArePreserved(unittest.TestCase):
    """Blank lines inside a string literal must survive preprocessing.

    Blank lines are stripped from Python cells because a blank line inside an
    indented block ends that block, so `class C:` followed by an empty line
    silently loses every method after it. The strip was applied to the whole
    cell, which also deleted blank lines *inside* triple-quoted strings, so a
    cell holding a file body, a SQL script or a text template produced different
    content than it was given -- the same damage as a file losing its blank
    lines. The original code carried the comment "(are we sure about this? test
    this)"; these tests are that test.
    """

    def _preprocess(self, code, *, active_line_detection):
        with patch.dict(
            os.environ, {"INTERPRETER_ACTIVE_LINE_DETECTION": active_line_detection}
        ):
            return preprocess_python(code)

    def test_blank_line_in_triple_quoted_string_survives(self):
        """A blank line between two lines of text stays in the string.

        The reported symptom was a triple-quoted A/blank/B literal arriving
        as a two-line string.
        """
        source = 's = """A\n\nB"""\n'
        for flag in ("true", "false"):
            with self.subTest(active_line_detection=flag):
                out = self._preprocess(source, active_line_detection=flag)
                self.assertEqual(_string_values(out), ["A\n\nB"])

    def test_whitespace_only_line_keeps_its_exact_content(self):
        """A line of only spaces inside a string is preserved verbatim.

        Not just kept as an empty line: the spaces are part of the value, so an
        indented block inside a string literal survives. Reported as
        'L1\\nL3' where 'L1\\n   \\nL3' was sent.
        """
        source = 't = """L1\n   \nL3"""\n'
        for flag in ("true", "false"):
            with self.subTest(active_line_detection=flag):
                out = self._preprocess(source, active_line_detection=flag)
                self.assertEqual(_string_values(out), ["L1\n   \nL3"])

    def test_trailing_blank_line_before_closing_quote_survives(self):
        """A blank line immediately before the closing quotes is kept.

        The awkward position, and the one most likely to be treated as
        insignificant whitespace during any line-level cleanup.
        """
        source = 'u = """X\n\n"""\n'
        for flag in ("true", "false"):
            with self.subTest(active_line_detection=flag):
                out = self._preprocess(source, active_line_detection=flag)
                self.assertEqual(_string_values(out), ["X\n\n"])

    def test_sql_like_template_is_untouched(self):
        """A blank line separating SQL clauses survives.

        The realistic case: a generated script held in a Python cell, where a
        dropped blank line changes the script that gets written out.
        """
        source = 'sql = """SELECT a\n\nFROM t\n\nWHERE a = 1"""\n'
        out = self._preprocess(source, active_line_detection="false")
        self.assertEqual(_string_values(out), ["SELECT a\n\nFROM t\n\nWHERE a = 1"])

    def test_docstring_blank_lines_survive(self):
        """A function docstring keeps its paragraph breaks.

        The docstring is a real multi-line string, so it is affected the same way
        a module-level one is; it is also the most common way to hit this.
        """
        source = 'def f():\n    """Doc.\n\n    More."""\n    return 1\n'
        out = self._preprocess(source, active_line_detection="false")
        self.assertIn('"""Doc.\n\n    More."""', out)

    def test_escaped_quotes_do_not_confuse_string_detection(self):
        """A line containing escaped triple quotes does not hide a real string.

        A hand-rolled scan that toggles state whenever a line contains three
        double quotes reads the escaped quotes in `he said \"\"\" ok` as
        opening a string, and would then treat the blank lines in the literal
        that follows as code. Detection is by tokenizer here, so the
        distinction is read rather than guessed.
        """
        source = 'q = "he said \\"\\"\\" ok"\ns = """A\n\nB"""\n'
        for flag in ("true", "false"):
            with self.subTest(active_line_detection=flag):
                out = self._preprocess(source, active_line_detection=flag)
                self.assertEqual(_string_values(out)[-1], "A\n\nB")

    def test_comment_mentioning_quotes_does_not_confuse_detection(self):
        """A `#` comment containing quotes is not a string delimiter."""
        source = 'h = "# not a comment"\ns = """A\n\nB"""\n'
        out = self._preprocess(source, active_line_detection="false")
        self.assertEqual(_string_values(out)[-1], "A\n\nB")


class TestBlankLinesOutsideStringsAreStillRemoved(unittest.TestCase):
    """The strip the string fix protects must keep doing its actual job.

    Without it, a blank line inside an indented block terminates the block and
    the code after it is silently dropped -- no error, just missing definitions.
    These guard the original behaviour so making the strip string-aware cannot
    regress into "leave blank lines alone".
    """

    def _preprocess(self, code):
        with patch.dict(os.environ, {"INTERPRETER_ACTIVE_LINE_DETECTION": "false"}):
            return preprocess_python(code)

    def test_blank_line_after_class_statement_does_not_break_the_body(self):
        """`class C:` then a blank line keeps the method that follows."""
        source = 'class C:\n\n    def m(self):\n\n        return "ok"\n'
        out = self._preprocess(source)
        self.assertNotIn("\n\n", out)
        self.assertIn("def m(self):", out)

    def test_blank_line_inside_function_body_does_not_break_the_body(self):
        """A blank line between statements keeps the rest of the function."""
        source = 'def f():\n    x = 1\n\n    return x + 1\n'
        out = self._preprocess(source)
        self.assertIn("return x + 1", out)
        self.assertNotIn("\n\n", out)

    def test_unparseable_code_still_gets_blank_lines_removed(self):
        """Code the tokenizer rejects falls back to the previous behaviour.

        A cell that cannot be tokenized is about to fail in the kernel anyway;
        the fallback keeps the old behaviour rather than inventing a guess about
        which of its lines were meant to be string content.
        """
        source = 'def f(:\n\n    pass\n'
        out = self._preprocess(source)
        self.assertNotIn("\n\n", out)


if __name__ == "__main__":
    unittest.main()