import os
import re
import shutil
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.utils import commit_info, system_debug_info


def _run_git(*args, cwd):
    """Run git for test setup with repository-location overrides removed.

    A developer shell may export `GIT_DIR`/`GIT_WORK_TREE` for another
    checkout; setup must operate on the repository the test names, not that one.
    """
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=commit_info._git_env(),
        capture_output=True,
        text=True,
        check=True,
    )


def _fake_git(short_sha, status=""):
    """Patch the git runner so it answers rev-parse and status without a checkout."""

    def run(args, cwd):
        if args[0] == "rev-parse":
            return short_sha
        if args[0] == "status":
            return status
        raise AssertionError(f"unexpected git call: {args}")

    return mock.patch.object(commit_info, "_git", side_effect=run)


def test_get_commit_info_reports_the_short_sha_and_a_clean_tree(tmp_path):
    """A clean checkout reports the short SHA with `dirty` false.

    The SHA is what a bug report needs; claiming the tree is modified when it
    is not would send maintainers looking for local changes that do not exist.
    """
    (tmp_path / ".git").mkdir()
    with _fake_git("abc1234", ""):
        info = commit_info.get_commit_info(tmp_path)

    assert info == commit_info.CommitInfo("abc1234", False)


def test_get_commit_info_flags_modified_tracked_files(tmp_path):
    """A checkout with edited tracked files is reported as dirty.

    `%info` and `--version` must warn that the reported SHA does not describe
    exactly the code that ran.
    """
    (tmp_path / ".git").mkdir()
    with _fake_git("abc1234", " M interpreter/core/core.py"):
        info = commit_info.get_commit_info(tmp_path)

    assert info == commit_info.CommitInfo("abc1234", True)


def test_get_commit_info_finds_the_repo_root_from_the_module_path(tmp_path, monkeypatch):
    """With no explicit root, the repository is found by walking up from this file.

    Editable installs run from inside the checkout, so the module path is the
    only hint available at runtime.
    """
    nested = tmp_path / "interpreter" / "core" / "utils"
    nested.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(commit_info, "__file__", str(nested / "commit_info.py"))

    with _fake_git("abc1234", ""):
        info = commit_info.get_commit_info()

    assert info == commit_info.CommitInfo("abc1234", False)


def test_find_repo_root_ignores_an_enclosing_application_repository(
    tmp_path, monkeypatch
):
    """A wheel installed in an app's .venv must not report the app's SHA.

    The walk from site-packages reaches the application's own `.git`, but that
    commit has nothing to do with Open Interpreter; labelling a report with it
    would send maintainers to the wrong repository entirely.
    """
    site_packages = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    module_dir = site_packages / "interpreter" / "core" / "utils"
    module_dir.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(commit_info, "__file__", str(module_dir / "commit_info.py"))

    assert commit_info._find_repo_root() is None


def test_get_commit_info_returns_none_outside_a_repository(tmp_path, monkeypatch):
    """A wheel or sdist install has no `.git`, so no commit is reported.

    Printing a stale or invented SHA would be worse than printing nothing.
    """
    monkeypatch.setattr(commit_info, "_find_repo_root", lambda: None)

    assert commit_info.get_commit_info() is None


def test_find_repo_root_returns_none_when_no_parent_has_git(tmp_path, monkeypatch):
    """A module that lives outside any checkout resolves no root at all."""
    monkeypatch.setattr(commit_info, "__file__", str(tmp_path / "commit_info.py"))
    assert commit_info._find_repo_root() is None


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("git"),
        subprocess.TimeoutExpired(cmd="git", timeout=5),
        subprocess.CalledProcessError(1, "git"),
    ],
)
def test_get_commit_info_returns_none_when_git_cannot_answer(tmp_path, error):
    """Missing git, a timeout, or a failing git command degrade to `unknown`.

    `--version` and `%info` must still print on machines without git rather
    than crash the CLI.
    """
    (tmp_path / ".git").mkdir()
    with mock.patch.object(commit_info, "_git", side_effect=error):
        assert commit_info.get_commit_info(tmp_path) is None


def test_format_version_line_appends_the_commit():
    """`--version` names the exact revision, including a dirty marker."""
    line = commit_info.format_version_line(
        "0.4.3", "Developer Preview", commit_info.CommitInfo("abc1234", False)
    )
    assert line == "Open Interpreter 0.4.3 Developer Preview (abc1234)"

    dirty = commit_info.format_version_line(
        "0.4.3", "Developer Preview", commit_info.CommitInfo("abc1234", True)
    )
    assert dirty == "Open Interpreter 0.4.3 Developer Preview (abc1234 dirty)"


def test_format_version_line_omits_the_commit_when_unknown():
    """Without a commit, the line keeps the historical wording unchanged."""
    assert (
        commit_info.format_version_line("0.4.3", "Developer Preview", None)
        == "Open Interpreter 0.4.3 Developer Preview"
    )


def test_system_info_reports_the_commit():
    """The `%info` report carries a commit line so reports are attributable."""
    interpreter = SimpleNamespace(
        offline=False,
        llm=SimpleNamespace(
            api_base=None,
            supports_vision=False,
            model="test-model",
            supports_functions=True,
            context_window=8000,
            max_tokens=1000,
        ),
        messages=[],
        system_message="test",
        auto_run=True,
        computer=SimpleNamespace(import_computer_api=False),
    )
    with mock.patch.object(
        system_debug_info, "get_commit_info", return_value=commit_info.CommitInfo("abc1234", True)
    ):
        with mock.patch("builtins.print") as fake_print:
            system_debug_info.system_info(interpreter)

    report = fake_print.call_args.args[0]
    assert "Commit: abc1234 (dirty)" in report


def test_system_info_reports_unknown_without_a_commit():
    """`%info` says `unknown` rather than omitting the line on non-git installs."""
    interpreter = SimpleNamespace(
        offline=False,
        llm=SimpleNamespace(
            api_base=None,
            supports_vision=False,
            model="test-model",
            supports_functions=True,
            context_window=8000,
            max_tokens=1000,
        ),
        messages=[],
        system_message="test",
        auto_run=True,
        computer=SimpleNamespace(import_computer_api=False),
    )
    with mock.patch.object(system_debug_info, "get_commit_info", return_value=None):
        with mock.patch("builtins.print") as fake_print:
            system_debug_info.system_info(interpreter)

    report = fake_print.call_args.args[0]
    assert "Commit: unknown" in report


def test_get_commit_info_works_against_the_real_checkout():
    """The real runner shells out to git and reports this repository's SHA.

    The other tests patch `_git`, so without this one the subprocess call and
    its argument handling would never run in CI.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    if commit_info._find_repo_root() is None:
        pytest.skip("not running from a git checkout")
    info = commit_info.get_commit_info()
    assert info is not None
    # `git rev-parse --short` honours core.abbrev, which allows four
    # characters, so do not demand the usual seven here.
    assert re.fullmatch(r"[0-9a-f]{4,40}", info.short_sha)


def test_git_runner_tolerates_undecodable_filename_bytes(tmp_path):
    """A tracked filename with bytes outside UTF-8 must not crash diagnostics.

    With `core.quotePath=false` git prints the raw bytes, and a strict decoder
    raises UnicodeDecodeError before `--version` or `%info` can print.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not installed")

    _run_git("init", "-q", cwd=tmp_path)
    _run_git("config", "core.quotePath", "false", cwd=tmp_path)
    name = b"\xff-undecodable"
    try:
        fd = os.open(
            os.path.join(os.fsencode(tmp_path), name), os.O_CREAT | os.O_WRONLY, 0o644
        )
    except (OSError, UnicodeError) as error:
        # macOS rejects the raw bytes and Windows' fsdecode refuses to turn
        # them back into a path; either way the scenario cannot be staged.
        pytest.skip(f"filesystem cannot store such a filename: {error}")
    os.close(fd)
    _run_git("add", ".", cwd=tmp_path)

    status = commit_info._git(
        ["status", "--porcelain", "--untracked-files=no"], tmp_path
    )

    assert "\ufffd" in status


def test_git_runner_ignores_repository_environment_overrides(tmp_path, monkeypatch):
    """Exported git variables must not redirect the reported commit.

    A shell that exports `GIT_DIR`/`GIT_WORK_TREE` for another checkout would
    otherwise make `--version` and `%info` report that checkout's SHA, which is
    exactly the confusion the commit line exists to prevent.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    root = commit_info._find_repo_root()
    if root is None:
        pytest.skip("not running from a git checkout")

    decoy = tmp_path / "decoy"
    decoy.mkdir()
    _run_git("init", "-q", cwd=decoy)
    _run_git(
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=Test",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "decoy",
        cwd=decoy,
    )
    decoy_sha = _run_git("rev-parse", "--short", "HEAD", cwd=decoy).stdout.strip()
    ours = _run_git("rev-parse", "--short", "HEAD", cwd=root).stdout.strip()
    assert decoy_sha != ours, "the decoy repository must have a distinct commit"

    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(decoy))

    # Confirm the override really would redirect an unscrubbed git call, so
    # this test fails if the environment scrubbing is removed.
    redirected = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert redirected == decoy_sha

    assert commit_info._git(["rev-parse", "--short", "HEAD"], root) == ours
