import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from interpreter.core.tools.file_edit import (
    EDIT_LANGUAGES,
    _comby_rewritten_source,
    _poke_prepare_script,
    _reject_empty_replace_output,
    _split_comby_templates,
    _validate_target,
    run_edit,
    run_gawk,
    run_jq,
    run_yq,
    run_patch,
    run_poke,
    run_comby,
    run_sed,
    run_write,
)


class TestEditLanguagesRegistry(unittest.TestCase):
    def test_expected_languages_registered(self):
        self.assertIn("comby", EDIT_LANGUAGES)
        self.assertIn("patch", EDIT_LANGUAGES)
        self.assertNotIn("perl", EDIT_LANGUAGES)
        self.assertNotIn("ed", EDIT_LANGUAGES)


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


class TestCombyJson(unittest.TestCase):
    def test_rewritten_source_from_json_object(self):
        payload = json.dumps({"rewritten_source": "value = 2\n"})
        out = _comby_rewritten_source(payload.encode())
        self.assertEqual(out, b"value = 2\n")

    def test_rewritten_source_from_json_lines(self):
        payload = json.dumps({"rewritten_source": "value = 2\n"})
        out = _comby_rewritten_source((payload + "\n").encode())
        self.assertEqual(out, b"value = 2\n")


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


class TestRunWrite(unittest.TestCase):
    def test_write_then_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "new.txt")
            self.assertTrue(run_write(target, "alpha\n").startswith("Wrote"))
            with self.assertRaises(FileExistsError):
                run_write(target, "beta\n")


@unittest.skipUnless(shutil.which("sed"), "sed not installed")
class TestRunSed(unittest.TestCase):
    def test_run_sed_substitute(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.txt")
            run_write(target, "foo bar\n")
            run_sed(target, "s/foo/baz/")
            self.assertEqual(open(target, encoding="utf-8").read(), "baz bar\n")

    def test_run_sed_multiline_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.txt")
            run_write(target, "aaa\nbbb\n")
            run_sed(target, "s/aaa/AAA/\ns/bbb/BBB/\n")
            self.assertEqual(open(target, encoding="utf-8").read(), "AAA\nBBB\n")


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

    def test_run_jq_multiline_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "data.json")
            run_write(target, '{"name":"x","n":1}\n')
            run_jq(
                target,
                "{\n  name: .name,\n  n: (.n + 1)\n}\n",
            )
            data = json.loads(open(target, encoding="utf-8").read())
            self.assertEqual(data["name"], "x")
            self.assertEqual(data["n"], 2)

    def test_run_jq_rejects_empty_filter_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "data.json")
            run_write(target, '{"a":1}\n')
            with self.assertRaises(RuntimeError) as ctx:
                run_jq(target, "empty")
            self.assertIn("0-byte", str(ctx.exception).lower())
            self.assertEqual(Path(target).read_text(encoding="utf-8"), '{"a":1}\n')


class TestRejectEmptyReplaceOutput(unittest.TestCase):
    def test_rejects_empty_when_original_had_content(self):
        with self.assertRaises(RuntimeError) as ctx:
            _reject_empty_replace_output("gawk", b"", 12)
        self.assertIn("0-byte", str(ctx.exception).lower())

    def test_allows_empty_when_original_was_empty(self):
        _reject_empty_replace_output("gawk", b"", 0)


@unittest.skipUnless(shutil.which("gawk"), "gawk not installed")
class TestRunGawk(unittest.TestCase):
    def test_run_gawk_multiline_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "lines.txt")
            run_write(target, "hello world\n")
            run_gawk(
                target,
                "{\n  gsub(/world/, \"earth\")\n  print\n}\n",
            )
            self.assertEqual(open(target, encoding="utf-8").read(), "hello earth\n")

    def test_run_gawk_rejects_program_with_no_print(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "lines.txt")
            Path(target).write_text("hello\n", encoding="utf-8")
            with self.assertRaises(RuntimeError) as ctx:
                run_gawk(target, "{ }\n")
            self.assertIn("0-byte", str(ctx.exception).lower())
            self.assertEqual(Path(target).read_text(encoding="utf-8"), "hello\n")


@unittest.skipUnless(shutil.which("yq"), "yq not installed")
class TestRunYq(unittest.TestCase):
    def test_run_yq_multiline_expression(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "data.json")
            run_write(target, '{"value": 1}\n')
            run_yq(
                target,
                ".value = (\n  .value + 1\n)\n",
            )
            data = json.loads(open(target, encoding="utf-8").read())
            self.assertEqual(data["value"], 2)


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


class TestPokeScript(unittest.TestCase):
    def test_prepare_script_file_and_quit(self):
        script = _poke_prepare_script("uint8 @ 0#B = 0x58", "/tmp/demo.bin")
        self.assertIn(".file /tmp/demo.bin\n", script)
        self.assertIn("uint8 @ 0#B = 0x58", script)
        self.assertTrue(script.rstrip().endswith(".quit"))

    def test_prepare_script_skips_file_if_present(self):
        script = _poke_prepare_script(".file other.bin\nbyte @ 0#B", "/tmp/demo.bin")
        self.assertEqual(script.count(".file"), 1)
        self.assertIn("other.bin", script)


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
