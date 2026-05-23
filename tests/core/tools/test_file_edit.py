import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from interpreter.core.tools.file_edit import (
    EDIT_LANGUAGES,
    _ed_output_is_error,
    _format_ed_failure,
    _poke_dot_file_arg,
    _split_comby_templates,
    _validate_target,
    run_edit,
    run_ed,
    run_jq,
    run_patch,
    run_poke,
    run_sed,
    run_write,
)


class TestEditLanguagesRegistry(unittest.TestCase):
    def test_expected_languages_registered(self):
        self.assertIn("comby", EDIT_LANGUAGES)
        self.assertIn("patch", EDIT_LANGUAGES)
        self.assertNotIn("perl", EDIT_LANGUAGES)


class TestValidateTarget(unittest.TestCase):
    def test_requires_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            rel = "demo.txt"
            with self.assertRaises(ValueError) as ctx:
                _validate_target(rel, must_exist=False)
            self.assertIn("absolute", str(ctx.exception).lower())

    def test_write_rejects_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "exists.txt")
            Path(target).write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _validate_target(target, must_exist=False)

    def test_edit_requires_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "missing.txt")
            with self.assertRaises(FileNotFoundError):
                _validate_target(target, must_exist=True)


class TestCombyHelpers(unittest.TestCase):
    def test_split_comby_multiline_separator(self):
        match, rewrite = _split_comby_templates(":[x] = 1\n---\n:[x] = 2")
        self.assertEqual(match, ":[x] = 1")
        self.assertEqual(rewrite, ":[x] = 2")

    def test_split_comby_two_lines(self):
        match, rewrite = _split_comby_templates("foo\nbar")
        self.assertEqual(match, "foo")
        self.assertEqual(rewrite, "bar")

    def test_split_comby_requires_two_parts(self):
        with self.assertRaises(ValueError):
            _split_comby_templates("only_one_line")


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


class TestRunWrite(unittest.TestCase):
    def test_write_then_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "new.txt")
            self.assertTrue(run_write(target, "alpha\n").startswith("Wrote"))
            with self.assertRaises(FileExistsError):
                run_write(target, "beta\n")


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


@unittest.skipUnless(shutil.which("sed"), "sed not installed")
class TestRunSed(unittest.TestCase):
    def test_run_sed_substitute(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.txt")
            run_write(target, "foo bar\n")
            run_sed(target, "s/foo/baz/")
            self.assertEqual(open(target, encoding="utf-8").read(), "baz bar\n")


@unittest.skipUnless(shutil.which("jq"), "jq not installed")
class TestRunJq(unittest.TestCase):
    def test_run_jq_transform_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "data.json")
            run_write(target, '{"a":1,"b":2}\n')
            run_jq(target, "{a, b: (.b + 1)}")
            data = json.loads(open(target, encoding="utf-8").read())
            self.assertEqual(data["a"], 1)
            self.assertEqual(data["b"], 3)


@unittest.skipUnless(shutil.which("patch"), "patch not installed")
class TestRunPatch(unittest.TestCase):
    def test_run_patch_unified_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.txt")
            run_write(target, "foo\nbar\n")
            diff = (
                f"--- {os.path.basename(target)}\n"
                f"+++ {os.path.basename(target)}\n"
                "@@ -1,2 +1,2 @@\n"
                "-foo\n"
                "+baz\n"
                " bar\n"
            )
            run_patch(target, diff)
            self.assertEqual(open(target, encoding="utf-8").read(), "baz\nbar\n")


@unittest.skipUnless(shutil.which("comby"), "comby not installed")
class TestRunComby(unittest.TestCase):
    def test_run_comby_stdin_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "code.py")
            run_write(target, "value = 1\n")
            run_comby(target, "value = 1\n---\nvalue = 2\n")
            self.assertEqual(open(target, encoding="utf-8").read(), "value = 2\n")


class TestPokePathQuoting(unittest.TestCase):
    def test_dot_file_unquoted_without_spaces(self):
        self.assertEqual(
            _poke_dot_file_arg("/tmp/demo.bin"),
            "/tmp/demo.bin",
        )

    def test_dot_file_quoted_when_path_has_spaces(self):
        self.assertEqual(
            _poke_dot_file_arg("/tmp/my dir/demo.bin"),
            '"/tmp/my dir/demo.bin"',
        )


@unittest.skipUnless(shutil.which("poke"), "poke not installed")
class TestRunPokeEdit(unittest.TestCase):
    def test_run_poke_script_includes_quit_and_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.bin")
            Path(target).write_bytes(b"AAAA")
            run_poke(target, "uint8 @ 0#B = 0x58")
            self.assertEqual(Path(target).read_bytes()[0], ord("X"))


class TestRunEditDispatch(unittest.TestCase):
    def test_unknown_language_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "x.txt")
            with self.assertRaises(ValueError) as ctx:
                run_edit("nosuch", "code", target)
            self.assertIn("nosuch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
