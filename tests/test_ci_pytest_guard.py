"""The CI JUnit guard must reject all-skip and required-runtime-skip runs."""

import importlib.util
from pathlib import Path

_GUARD_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "check_pytest_counts.py"
)
_SPEC = importlib.util.spec_from_file_location("check_pytest_counts", _GUARD_PATH)
guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guard)


def _write(path, body):
    path.write_text(f"<testsuite>{body}</testsuite>")
    return str(path)


def test_guard_accepts_a_passing_run(tmp_path, monkeypatch, capsys):
    """A normal run with enough passing tests exits zero."""
    path = _write(tmp_path / "ok.xml", '<testcase classname="t" name="a"/>')
    monkeypatch.setattr("sys.argv", ["guard", path, "1"])

    assert guard.main() == 0
    assert "pytest guard: 1 passed" in capsys.readouterr().out


def test_guard_rejects_a_required_runtime_skip(tmp_path, monkeypatch, capsys):
    """A skip reason the Linux workflow must prevent exits nonzero."""
    path = _write(
        tmp_path / "skip.xml",
        '<testcase classname="t" name="a"><skipped message="irb not installed"/></testcase>',
    )
    monkeypatch.setattr("sys.argv", ["guard", path, "1", "irb not installed"])

    assert guard.main() == 1
    assert "::error::" in capsys.readouterr().out


def test_guard_rejects_too_few_passing_tests(tmp_path, monkeypatch, capsys):
    """An all-skip selection is caught by the minimum-passed floor."""
    path = _write(
        tmp_path / "all-skip.xml",
        '<testcase classname="t" name="a"><skipped message="no key"/></testcase>',
    )
    monkeypatch.setattr("sys.argv", ["guard", path, "1"])

    assert guard.main() == 1
    assert "only 0 tests passed" in capsys.readouterr().out


def test_guard_counts_xfails_separately(tmp_path, monkeypatch, capsys):
    """xfails are reported separately and do not count as passed tests."""
    path = _write(
        tmp_path / "mixed.xml",
        '<testcase classname="t" name="a"/>'
        '<testcase classname="t" name="b">'
        '<skipped type="pytest.xfail" message="known bug"/>'
        "</testcase>",
    )
    monkeypatch.setattr("sys.argv", ["guard", path, "1"])

    assert guard.main() == 0
    assert "1 passed, 0 skipped, 1 xfailed, 0 failed" in capsys.readouterr().out
