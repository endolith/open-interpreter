import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from interpreter.core.tools.file_edit import (
    EDIT_LANGUAGES,
    _assert_mikefarah_yq,
    _comby_rewritten_source,
    _poke_prepare_script,
    _reject_wrong_os_path,
    _split_comby_templates,
    _validate_target,
    _write_temp_script,
    _yq_eval_argv_candidates,
    dry_run_edit,
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

    def test_rejects_git_bash_drive_path_on_windows(self):
        """A Git Bash style /c/... path must fail on Windows before it creates a literal 'c' folder."""
        with mock.patch("platform.system", return_value="Windows"):
            with self.assertRaises(ValueError) as ctx:
                _validate_target("/c/Users/test/file.txt", must_exist=True)
            self.assertIn("Git Bash", str(ctx.exception))

    def test_rejects_drive_relative_path_on_windows(self):
        """A rooted path with no drive letter resolves against the current drive, so reject it."""
        with mock.patch("platform.system", return_value="Windows"):
            with self.assertRaises(ValueError) as ctx:
                _validate_target("/Users/test/file.txt", must_exist=True)
            self.assertIn("drive letter", str(ctx.exception))

    def test_rejects_windows_path_on_posix(self):
        """A C:/... path on Mac/Linux must fail with an OS-specific message, not the generic absolute-path one."""
        with mock.patch("platform.system", return_value="Linux"):
            with self.assertRaises(ValueError) as ctx:
                _validate_target("C:/Users/test/file.txt", must_exist=True)
            self.assertIn("not Windows", str(ctx.exception))

    def test_accepts_well_formed_paths_for_each_os(self):
        """Well-formed paths must pass the OS-shape check on their own OS."""
        with mock.patch("platform.system", return_value="Windows"):
            _reject_wrong_os_path("C:/Users/test/file.txt")
            _reject_wrong_os_path("C:\\Users\\test\\file.txt")
            _reject_wrong_os_path("\\\\server\\share\\file.txt")
        with mock.patch("platform.system", return_value="Linux"):
            _reject_wrong_os_path("/home/test/file.txt")


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
    def test_write_then_replace(self):
        """write creates, and replaces an existing file verbatim.

        Overwriting used to be refused, which taught models a worse habit:
        write, get blocked, delete the file, write again. A replacement now
        previews as a diff upstream, so by the time this runs it has been
        reviewed twice.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "new.txt")
            self.assertTrue(run_write(target, "alpha\n").startswith("Wrote"))
            self.assertTrue(run_write(target, "beta\n").startswith("Wrote"))
            self.assertEqual(open(target, encoding="utf-8").read(), "beta\n")


class TestDryRunWrite(unittest.TestCase):
    def test_write_overwrite_previews_a_diff(self):
        """A write that would replace a file previews the resulting diff."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.txt")
            run_write(target, "old line\n")
            preview = dry_run_edit("write", "new line\n", target)
            self.assertTrue(preview["ok"])
            self.assertIn("-old line", preview["output"])
            self.assertIn("+new line", preview["output"])

    def test_write_new_file_has_no_preview(self):
        """A brand-new file has nothing to diff against: no preview, no gate."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "missing.txt")
            self.assertIsNone(dry_run_edit("write", "hello\n", target))

    def test_write_identical_content_is_no_change(self):
        """Rewriting byte-identical content is flagged, not diffed."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.txt")
            run_write(target, "same\n")
            preview = dry_run_edit("write", "same\n", target)
            self.assertTrue(preview["ok"])
            self.assertTrue(preview.get("no_change"))


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

    def test_run_sed_does_not_use_inplace_flag(self):
        """sed -i puts temp files in cwd; on Windows that breaks cross-drive targets."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.txt")
            run_write(target, "foo\n")
            with mock.patch(
                "interpreter.core.tools.file_edit.subprocess.run"
            ) as run_mock:
                completed = mock.Mock(returncode=0, stdout=b"bar\n", stderr=b"")
                run_mock.return_value = completed
                with mock.patch(
                    "interpreter.core.tools.file_edit._atomic_replace_from_stdout"
                ) as replace_mock:
                    run_sed(target, "s/foo/bar/")
            args = run_mock.call_args[0][0]
            self.assertNotIn("-i", args)
            replace_mock.assert_called_once_with(target, b"bar\n")

    def test_run_sed_no_match_warns_and_leaves_file_untouched(self):
        """A non-matching sed pattern must warn instead of reporting plain OK."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.txt")
            run_write(target, "foo\n")
            before_mtime = os.path.getmtime(target)
            result = run_sed(target, "s/does_not_match/xxx/")
            self.assertIn("no changes", result)
            self.assertEqual(open(target, encoding="utf-8").read(), "foo\n")
            self.assertEqual(os.path.getmtime(target), before_mtime)

    def test_run_sed_preserves_executable_bit(self):
        """Editing a script must not clear its executable bit."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.sh")
            run_write(target, "foo\n")
            os.chmod(target, 0o755)
            run_sed(target, "s/foo/bar/")
            self.assertEqual(open(target, encoding="utf-8").read(), "bar\n")
            self.assertTrue(os.access(target, os.X_OK))
            self.assertEqual(oct(os.stat(target).st_mode & 0o777), "0o755")


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

    def test_write_temp_script_utf8_non_ascii(self):
        path = _write_temp_script('.label = "✅"', ".jq")
        try:
            self.assertEqual(open(path, "rb").read(), b'.label = "\xe2\x9c\x85"\n')
        finally:
            os.remove(path)

    def test_run_jq_unicode_in_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "data.json")
            run_write(target, '{"label":"old"}\n')
            run_jq(target, '.label = "✅"')
            data = json.loads(open(target, encoding="utf-8").read())
            self.assertEqual(data["label"], "✅")

    def test_run_jq_uses_atomic_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "data.json")
            run_write(target, '{"a":1}\n')
            with mock.patch(
                "interpreter.core.tools.file_edit.subprocess.run"
            ) as run_mock:
                run_mock.return_value = mock.Mock(
                    returncode=0, stdout=b'{"a":2}\n', stderr=b""
                )
                with mock.patch(
                    "interpreter.core.tools.file_edit._atomic_replace_from_stdout"
                ) as replace_mock:
                    run_jq(target, ".a = 2")
            replace_mock.assert_called_once_with(target, b'{"a":2}\n')


@unittest.skipUnless(shutil.which("gawk"), "gawk not installed")
class TestRunGawk(unittest.TestCase):
    def test_run_gawk_multiline_program(self):
        """A per-line gawk program must rewrite the file with its stdout."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "lines.txt")
            run_write(target, "hello world\n")
            run_gawk(
                target,
                "{\n  gsub(/world/, \"earth\")\n  print\n}\n",
            )
            self.assertEqual(open(target, encoding="utf-8").read(), "hello earth\n")

    def test_run_gawk_does_not_use_inplace_flag(self):
        """Stdout+atomic replace is deterministic; -i inplace wiped END-only files."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "lines.txt")
            run_write(target, "hello\n")
            with mock.patch(
                "interpreter.core.tools.file_edit.subprocess.run"
            ) as run_mock:
                run_mock.return_value = mock.Mock(
                    returncode=0, stdout=b"hello\n", stderr=b""
                )
                with mock.patch(
                    "interpreter.core.tools.file_edit._atomic_replace_from_stdout"
                ):
                    run_gawk(target, "{ print }")
            args = run_mock.call_args[0][0]
            self.assertNotIn("-i", args)
            self.assertNotIn("inplace", args)
            self.assertIn(target, args)

    def test_run_gawk_empty_output_does_not_wipe_file(self):
        """Empty gawk output for a non-empty file must abort instead of truncating."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "lines.txt")
            run_write(target, "a\nb\nc\n")
            with self.assertRaises(RuntimeError) as ctx:
                run_gawk(target, "END { print \"\" }")
            self.assertIn("no output", str(ctx.exception))
            self.assertEqual(open(target, encoding="utf-8").read(), "a\nb\nc\n")

    def test_run_gawk_no_changes_reports_notice(self):
        """Identical gawk output must report no-changes instead of plain OK."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "lines.txt")
            run_write(target, "hello\n")
            result = run_gawk(target, "{ print }")
            self.assertIn("no changes", result)
            self.assertEqual(open(target, encoding="utf-8").read(), "hello\n")


@unittest.skipUnless(shutil.which("yq"), "yq not installed")
class TestRunYq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yq = shutil.which("yq")
        if not yq:
            raise unittest.SkipTest("yq not installed")
        try:
            _assert_mikefarah_yq(yq)
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_yq_argv_uses_inline_expression_not_from_file(self):
        # Regression: -f / --from-file exited 0 but did not apply the expression.
        yq = shutil.which("yq")
        for args in _yq_eval_argv_candidates(yq, ".server.port = 9090", "/tmp/a.yaml"):
            self.assertNotIn("-f", args)
            self.assertNotIn("--from-file", args)
            self.assertIn(".server.port = 9090", args)

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

    def test_run_yq_yaml_in_place_not_stdout_only(self):
        # Regression: eval -i -f exited 0 but left the file unchanged (2026-05-24).
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "config.yaml")
            run_write(
                target,
                "server:\n  port: 8080\n  host: localhost\n",
            )
            self.assertEqual(run_yq(target, ".server.port = 9090"), "yq: OK")
            text = open(target, encoding="utf-8").read()
            self.assertIn("9090", text)
            self.assertNotIn("8080", text)

    def test_run_yq_chained_updates_on_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "app.yaml")
            run_write(target, "version: '1.0'\ndebug: false\n")
            run_yq(target, '.version = "2.0" | .debug = true')
            text = open(target, encoding="utf-8").read()
            self.assertIn("2.0", text)
            self.assertIn("true", text)

    def test_dry_run_yq_does_not_modify_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "config.yaml")
            original = "server:\n  port: 8080\n"
            run_write(target, original)
            preview = dry_run_edit("yq", ".server.port = 9090", target)
            self.assertTrue(preview["ok"])
            self.assertIn("9090", preview["output"])
            self.assertEqual(open(target, encoding="utf-8").read(), original)

    def test_run_yq_uses_atomic_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "config.yaml")
            run_write(target, "port: 8080\n")
            with mock.patch(
                "interpreter.core.tools.file_edit._run_yq_eval"
            ) as eval_mock:
                eval_mock.return_value = mock.Mock(
                    returncode=0, stdout=b"port: 9090\n", stderr=b""
                )
                with mock.patch(
                    "interpreter.core.tools.file_edit._atomic_replace_from_stdout"
                ) as replace_mock:
                    run_yq(target, ".port = 9090")
            replace_mock.assert_called_once_with(target, b"port: 9090\n")


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

    def test_run_patch_uses_target_parent_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.txt")
            run_write(target, "foo\n")
            with mock.patch(
                "interpreter.core.tools.file_edit.subprocess.run"
            ) as run_mock:
                run_mock.return_value = mock.Mock(
                    returncode=0, stdout=b"", stderr=b""
                )
                run_patch(
                    target,
                    f"--- {Path(target).name}\n+++ {Path(target).name}\n",
                )
            self.assertEqual(
                run_mock.call_args.kwargs["cwd"], str(Path(target).parent)
            )


@unittest.skipUnless(shutil.which("patch"), "patch not installed")
class TestPatchFailureDiagnostics(unittest.TestCase):
    """Failures must name the mistake, not just report that patch said no.

    Every case here is something a model actually submitted. GNU patch's own
    diagnostics ("malformed patch at line 11", "Only garbage was found") name
    the symptom rather than the cause, so the same diff gets re-submitted
    corrected in the wrong direction -- in one logged case the hunk header was
    "fixed" to match a number patch had itself invented.
    """

    def _target(self, tmp, body="import os\nimport sys\nimport json\n\n\ndef helper():\n    return 1\n"):
        target = os.path.join(tmp, "mod.py")
        run_write(target, body)
        return target

    def test_begin_patch_envelope_is_explained(self):
        """A '*** Begin Patch' envelope gets a message naming the envelope.

        This is another tool's patch format. patch cannot parse it and says only
        "Only garbage was found in the patch input", which reads as a corrupt
        diff rather than a wrapped one, so the model re-sends the same envelope.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = self._target(tmp)
            diff = (
                "*** Begin Patch\n"
                f"*** Update File: {os.path.basename(target)}\n"
                "@@ 4,2 +4,3 @@\n"
                " def helper():\n"
                "     return 1\n"
                "+    pass\n"
                "*** End Patch\n"
            )
            with self.assertRaises(RuntimeError) as caught:
                run_patch(target, diff)
            message = str(caught.exception)
            self.assertIn("Begin Patch", message)
            self.assertIn("bare unified diff", message)

    def test_hunk_without_trailing_context_is_explained(self):
        """A hunk ending in '+' mid-file is explained with the trailing-context rule.

        Verified against patch 2.8: a hunk whose final body line is an addition
        is rejected mid-file at any amount of leading context, and accepted once
        one unchanged context line follows it. patch reports only "malformed
        patch at line N", leaving the model nothing to act on.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = self._target(tmp)
            # Insert after "import json" (line 3) with no trailing context,
            # while the file clearly continues past line 3.
            diff = (
                f"--- {os.path.basename(target)}\n"
                f"+++ {os.path.basename(target)}\n"
                "@@ -1,3 +1,4 @@\n"
                " import os\n"
                " import sys\n"
                " import json\n"
                "+import re\n"
            )
            with self.assertRaises(RuntimeError) as caught:
                run_patch(target, diff)
            message = str(caught.exception)
            self.assertIn("fewer than two unchanged", message)
            self.assertIn("end-of-file", message)

    def test_wrong_context_does_not_get_the_trailing_context_hint(self):
        """A hunk aimed at the wrong text is told that, not blamed on context depth.

        The trailing-context hint is only correct when the context really is in
        the file. Given a hunk whose context matches nothing, naming context
        depth sends the caller off padding a hunk that can never apply -- a
        confident wrong answer is worse than none, because it gets acted on.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = self._target(tmp)
            diff = (
                f"--- {os.path.basename(target)}\n"
                f"+++ {os.path.basename(target)}\n"
                "@@ -1,4 +1,5 @@\n"
                " totally wrong line\n"
                " nothing like this\n"
                " nor this one\n"
                " \n"
                "+added line\n"
                "\n\n"
            )
            with self.assertRaises(RuntimeError) as caught:
                run_patch(target, diff)
            message = str(caught.exception)
            self.assertNotIn("fewer than two unchanged", message)
            self.assertIn("did not match the file", message)

    def test_failure_message_does_not_point_at_a_deleted_rej_file(self):
        """The message must not name a .rej file, because the file is removed.

        patch always reports 'saving rejects to file X.rej', and the artifact is
        cleaned up immediately afterwards. Passed through unmodified, the message
        points the user at a file that does not exist -- and at the same time
        discards the reject, which is the only side-by-side of the hunk against
        the file's real lines.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = self._target(tmp)
            diff = (
                f"--- {os.path.basename(target)}\n"
                f"+++ {os.path.basename(target)}\n"
                "@@ -1,4 +1,5 @@\n"
                " totally wrong line\n"
                " nothing like this\n"
                " nor this one\n"
                " \n"
                "+added line\n"
                "\n\n"
            )
            with self.assertRaises(RuntimeError) as caught:
                run_patch(target, diff)
            message = str(caught.exception)
            self.assertNotIn("saving rejects to file", message)
            self.assertFalse(
                (Path(tmp) / (os.path.basename(target) + ".rej")).exists()
            )
            # The reject's content is preserved in the message instead.
            self.assertIn("totally wrong line", message)


    def test_same_hunk_with_two_trailing_context_lines_applies(self):
        """Adding two trailing context lines makes the identical hunk apply.

        The counterpart to the diagnostic above: it proves the rule the message
        states is the real one, so the advice given to the model is correct
        rather than merely plausible. Measured on patch 2.8: 0 trailing context
        lines fails, 1 fails, 2 applies, 3 applies.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = self._target(tmp)
            diff = (
                f"--- {os.path.basename(target)}\n"
                f"+++ {os.path.basename(target)}\n"
                "@@ -1,5 +1,6 @@\n"
                " import os\n"
                " import sys\n"
                " import json\n"
                "+import re\n"
                " \n"
                " \n"
            )
            run_patch(target, diff)
            self.assertIn("import re", open(target, encoding="utf-8").read())

    def test_failure_leaves_no_orig_or_rej_beside_the_target(self):
        """A failed patch must not litter the user's directory with .orig/.rej.

        patch writes both as a debugging aid whenever a hunk fails. They land
        next to the file the user asked to edit, and a stale .rej reads as an
        unresolved edit on the next visit. A logged session had to delete both
        by hand.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = self._target(tmp)
            diff = (
                f"--- {os.path.basename(target)}\n"
                f"+++ {os.path.basename(target)}\n"
                "@@ -1,3 +1,4 @@\n"
                " totally different\n"
                " nothing like the file\n"
                " nor this third line\n"
                "+added\n"
            )
            with self.assertRaises(RuntimeError):
                run_patch(target, diff)
            leftovers = sorted(p.name for p in Path(tmp).iterdir())
            self.assertEqual(leftovers, ["mod.py"], f"unexpected leftovers: {leftovers}")

    def test_successful_patch_leaves_no_artifacts(self):
        """Success must be clean too -- no .orig in the user's directory."""
        with tempfile.TemporaryDirectory() as tmp:
            target = self._target(tmp)
            diff = (
                f"--- {os.path.basename(target)}\n"
                f"+++ {os.path.basename(target)}\n"
                "@@ -1,5 +1,6 @@\n"
                " import os\n"
                " import sys\n"
                " import json\n"
                "+import re\n"
                " \n"
                " \n"
            )
            run_patch(target, diff)
            leftovers = sorted(p.name for p in Path(tmp).iterdir())
            self.assertEqual(leftovers, ["mod.py"], f"unexpected leftovers: {leftovers}")


@unittest.skipUnless(shutil.which("sed"), "sed not installed")
class TestDryRunPreviewIsADiff(unittest.TestCase):
    """A dry run must show the change as a diff, not the whole rewritten file.

    Every language here except `patch` is dry-run *without* its in-place flag,
    so what the tool prints is the entire edited file. Re-reading a 4000-line
    file to find the one line that changed is the cost, and it grows with the
    file. These tests pin the diff shape, the context depth, and the two cases
    that must not be mistaken for a change: an edit that changes nothing, and a
    target whose bytes are not text.
    """

    def _file(self, tmp, name="mod.py", lines=40):
        target = os.path.join(tmp, name)
        run_write(target, "".join(f"line {i}\n" for i in range(1, lines + 1)))
        return target

    def test_preview_is_a_unified_diff_not_the_whole_file(self):
        """A one-line edit in a long file previews as a short diff.

        The size is the point: the preview must not scale with the file, or it
        still costs the reader the same scan it was meant to save.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = self._file(tmp, lines=4000)
            out = dry_run_edit("sed", "s/line 2000$/CHANGED/", target)["output"]
            self.assertTrue(out.startswith("--- mod.py"), out[:80])
            self.assertIn("@@ ", out)
            self.assertIn("-line 2000", out)
            self.assertIn("+CHANGED", out)
            self.assertLess(out.count("\n"), 25, f"preview should stay short, got {out.count(chr(10))} lines")

    def test_preview_shows_three_lines_of_context_each_side(self):
        """Enough surrounding context to locate the change, not the whole file.

        Three lines is the unified-diff convention and is what makes the hunk
        readable in isolation; one line would not show what the edit is
        operating on.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = self._file(tmp, lines=40)
            out = dry_run_edit("sed", "s/line 20$/CHANGED/", target)["output"]
            before = out.index("-line 20")
            for n in (19, 18, 17):
                self.assertIn(f" line {n}", out[:before], f"missing leading context line {n}")
            after = out.index("+CHANGED")
            for n in (21, 22, 23):
                self.assertIn(f" line {n}", out[after:], f"missing trailing context line {n}")

    def test_unchanged_trailing_line_is_not_reported_as_modified(self):
        """The final line must not show as -/+ when the edit did not touch it.

        Regression: the preview was built from stripped stdout, so the file's
        trailing newline was dropped and the last line came out as a
        changed-then-identical pair on every single edit.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = self._file(tmp, lines=12)
            out = dry_run_edit("sed", "s/line 5$/CHANGED/", target)["output"]
            self.assertNotIn("-line 12", out)
            self.assertNotIn("+line 12", out)

    def test_edit_that_changes_nothing_says_so_instead_of_dumping_the_file(self):
        """A no-op edit previews as one line, not the entire unchanged file.

        Falling back to raw output here would be the worst case of the old
        behaviour: the reader gets thousands of lines and no signal that nothing
        happened.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = self._file(tmp, lines=2000)
            out = dry_run_edit("sed", "s/no-such-text/other/", target)["output"]
            self.assertIn("no changes", out)
            self.assertLess(out.count("\n"), 3)

    def test_undecodable_target_reports_that_no_preview_is_available(self):
        """A non-text target must not produce a fabricated diff.

        difflib over replacement characters would report changes the tool never
        made, and the raw output would decode with U+FFFD, which reads as
        corruption rather than as a preview. Neither is useful, so the preview
        says it is unavailable. The edit itself is unaffected.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "blob.bin")
            with open(target, "wb") as f:
                f.write(b"\xff\xfe\x00\x01not text")
            result = dry_run_edit("sed", "s/not/text/", target)
            self.assertNotIn("@@ ", result["output"])
            self.assertNotIn("\ufffd", result["output"])
            self.assertIn("preview unavailable", result["output"])

    def test_diff_language_is_used_when_the_preview_is_a_diff(self):
        """The fence language must follow the content, not the edit language.

        Highlighting a diff as Python or YAML paints every +/- line as ordinary
        code, discarding the only cue that says which lines changed.
        """
        from interpreter.terminal_interface.terminal_interface import (
            _looks_like_unified_diff,
        )

        self.assertTrue(
            _looks_like_unified_diff("--- a.py\n+++ a.py\n@@ -1 +1 @@\n-x\n+y\n")
        )
        # A plain file body must not be mistaken for a diff.
        self.assertFalse(_looks_like_unified_diff("x = 1\ny = 2\n"))
        # An error message must not be mistaken for a diff either.
        self.assertFalse(_looks_like_unified_diff("sed: can't read f.txt\n"))


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

    def test_prepare_script_does_not_quote_a_path_with_spaces(self):
        """poke's .file takes the rest of the line literally.

        Quoting the path made the quotes part of the filename, so any target
        containing a space failed with 'error: opening "\"/tmp/a b/f.bin\""'.
        """
        script = _poke_prepare_script("uint8 @ 0#B = 0x58", "/tmp/a b/f.bin")
        self.assertIn(".file /tmp/a b/f.bin\n", script)
        self.assertNotIn('"', script.splitlines()[0])


@unittest.skipUnless(shutil.which("poke"), "poke not installed")
class TestRunPokeEdit(unittest.TestCase):
    def test_run_poke_script_includes_quit_and_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "demo.bin")
            Path(target).write_bytes(b"AAAA")
            run_poke(target, "uint8 @ 0#B = 0x58")
            self.assertEqual(Path(target).read_bytes()[0], ord("X"))

    def test_run_poke_edits_a_path_containing_spaces(self):
        """A spaced target must actually be patched, not just quoted.

        The .file line is the only place the path reaches poke, so this covers
        the whole path end to end: temp script, .file line, resulting bytes.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = os.path.join(tmp, "a dir", "nested")
            os.makedirs(directory)
            target = os.path.join(directory, "demo file.bin")
            Path(target).write_bytes(b"AAAA")
            run_poke(target, "uint8 @ 0#B = 0x58")
            self.assertEqual(Path(target).read_bytes(), b"XAAA")

    def test_run_poke_edits_a_path_containing_a_quote(self):
        """Quotes and backslashes in the name stay literal.

        Nothing goes through a shell and poke reads .file verbatim, so a name
        like this has to round-trip untouched.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, 'we"ird \\ name.bin')
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
