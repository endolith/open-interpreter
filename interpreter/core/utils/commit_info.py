"""Resolve the commit a checkout was built from, for bug reports.

Installs from different commits, or from `main`, `classic/develop`, or a feature
branch, are indistinguishable when the version string matches, which makes bug
reports hard to correlate with the code that produced them. The commit is
resolved at runtime rather than baked into package metadata so editable installs
and plain git checkouts work without a build hook, and sdist/wheel installs
simply report no commit.
"""

import os
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional


# Git honours these variables before it looks at the working directory, so a
# shell that exports them for another checkout would make `--version` report
# that checkout's commit instead of ours.
_REPO_LOCATION_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
)


def _git_env():
    """Return a copy of the environment without git repository overrides."""
    return {k: v for k, v in os.environ.items() if k not in _REPO_LOCATION_VARS}


class CommitInfo(NamedTuple):
    """A checkout's short commit SHA and whether tracked files are modified."""

    short_sha: str
    dirty: bool


def _find_repo_root() -> Optional[Path]:
    """Return the checkout this module belongs to, or None when it is installed."""
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        # `.git` is a directory in a normal checkout and a file in a linked
        # worktree, so test for existence rather than is_dir().
        if not (parent / ".git").exists():
            continue
        # A wheel installed into an application's .venv sits inside that
        # application, whose `.git` is above the site-packages directory.
        # Reporting the application's SHA as the Open Interpreter revision
        # would be worse than reporting nothing, so only accept a checkout
        # whose own `interpreter/` directory contains this module.
        if module_path.is_relative_to(parent / "interpreter"):
            return parent
        return None
    return None


def _git(args, cwd):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=_git_env(),
        capture_output=True,
        text=True,
        # `core.quotePath=false` makes git emit raw filename bytes, which are
        # not guaranteed to decode. Diagnostics must not die over a filename.
        errors="replace",
        timeout=5,
        check=True,
    ).stdout.strip()


def get_commit_info(repo_root: Optional[Path] = None) -> Optional[CommitInfo]:
    """Return the short SHA for `repo_root`, or None when git cannot answer.

    Untracked files are ignored when deciding whether the tree is dirty, so
    scratch files a developer left lying around do not label every report as
    coming from a modified build.
    """
    root = repo_root if repo_root is not None else _find_repo_root()
    if root is None:
        return None
    try:
        short_sha = _git(["rev-parse", "--short", "HEAD"], root)
        dirty = bool(_git(["status", "--porcelain", "--untracked-files=no"], root))
    except (OSError, subprocess.SubprocessError):
        # git missing, a timeout, or a non-repository directory all mean the
        # same thing to the caller: no commit to report.
        return None
    return CommitInfo(short_sha, dirty) if short_sha else None


def format_version_line(
    package_version, update_name: str, commit: Optional[CommitInfo]
) -> str:
    """Build the `--version` line, appending the commit when one is known."""
    line = f"Open Interpreter {package_version} {update_name}"
    if commit is None:
        return line
    suffix = " dirty" if commit.dirty else ""
    return f"{line} ({commit.short_sha}{suffix})"
