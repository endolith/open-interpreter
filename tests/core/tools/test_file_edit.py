import os
import shutil
import tempfile
import unittest

from interpreter.core.tools.file_edit import (
    _ed_output_is_error,
    _format_ed_failure,
    _split_comby_templates,
    run_ed,
    run_perl,
    run_write,
)


class TestCombyHelpers(unittest.TestCase):
    def test_split_comby_multiline_separator(self):
        match, rewrite = _split_comby_templates(":[x] = 1\n---\n:[x] = 2")
        self.assertEqual(match, ":[x] = 1")
        self.assertEqual(rewrite, ":[x] = 2")

    def test_split_comby_two_lines(self):
        match, rewrite = _split_comby_templates("foo\nbar")
        self.assertEqual(match, "foo")
        self.assertEqual(rewrite, "bar")


class TestEdHelpers(unittest.TestCase):
    def test_ed_output_is_error_question_marks_only(self):
        self.assertTrue(_ed_output_is_error("?\n?", ""))
        self.assertTrue(_ed_output_is_error("", "?\n"))
        self.assertFalse(_ed_output_is_error("", "15\n16\n"))

    def test_format_ed_failure_not_bare_question_mark(self):
        msg = _format_ed_failure("?\n?", "", 1, "2\ns/foo/bar/\nwq\n")
        self.assertNotEqual(msg.strip(), "?")
        self.assertIn("Script:", msg)
        self.assertIn("ed command failed", msg)


class TestRunEd(unittest.TestCase):
    def test_run_ed_success_and_failure_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.txt")
            run_write(target, "line one\nline two\n")
            run_ed(target, "2\ns/two/TOO/\nwq\n")
            self.assertIn("TOO", open(target, encoding="utf-8").read())

            with self.assertRaises(RuntimeError) as ctx:
                run_ed(target, "9\ns/nope/nope/\nwq\n")
            msg = str(ctx.exception)
            self.assertNotEqual(msg.strip(), "?")
            self.assertIn("Script:", msg)


@unittest.skipUnless(shutil.which("perl"), "perl not installed")
class TestRunPerl(unittest.TestCase):
    def test_run_perl_substitute(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.txt")
            run_write(target, "hello world\n")
            run_perl(target, r"s/hello/hi/")
            self.assertIn("hi world", open(target, encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()
