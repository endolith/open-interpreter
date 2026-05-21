import os
import tempfile
import unittest

from interpreter.core.tools.file_edit import (
    run_edit,
    run_write,
    validate_target,
)


class TestFileEdit(unittest.TestCase):
    def test_validate_target_requires_absolute(self):
        with self.assertRaises(ValueError):
            validate_target("relative.txt", must_exist=False)

    def test_write_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "new.txt")
            msg = run_write(path, "hello")
            self.assertIn("Wrote", msg)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "hello")

    def test_write_refuses_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "exists.txt")
            open(path, "w").close()
            with self.assertRaises(FileExistsError):
                run_write(path, "x")

    def test_run_edit_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "via_edit.txt")
            out = run_edit("write", "content", path)
            self.assertIn("Wrote", out)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "content")

    def test_jq_transform(self):
        import shutil

        if not shutil.which("jq"):
            self.skipTest("jq not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"a": 1}')
            run_edit("jq", ".a = 2", path)
            with open(path, encoding="utf-8") as f:
                self.assertIn("2", f.read())


if __name__ == "__main__":
    unittest.main()
