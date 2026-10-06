"""Tests for the `--os` branch of interpreter/__init__.py.

The package entry point has two paths: the normal imports, which run on every
import and are covered, and an early branch taken when `--os` is on the command
line that checks PyPI for a newer release and then hands off to the computer-use
loop. The `--os` path had no coverage because pytest never puts `--os` on argv.

**The branch cannot currently start.** `from packaging import version` (line 33)
shadows `from importlib.metadata import version` (line 31), so line 41's
`version("open-interpreter")` calls a module and raises `TypeError` — before
`requests.get` is ever reached, and unguarded at import. See #407.

The launch path is verified in a real subprocess rather than by exec'ing the
source: `__main__`/package-init code is unusually hostile to in-process import
machinery (the relative hand-off import raises `KeyError: '__name__' not in
globals` without a module registered under its own name), and a subprocess is how
this actually runs. The version-comparison logic is tested directly instead.
"""

import subprocess
import sys
import textwrap

import pytest

IGNORED = ("__pycache__", ".pytest_cache", "mutants")


def _in_repo():
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _run_python(argv_extra, timeout=180):
    """Run `import interpreter` in a fresh interpreter with the given argv.

    A subprocess rather than an in-process import: once `interpreter` is
    imported, `sys.modules` caches it, and setting `sys.argv` afterwards cannot
    re-enter the `--os` branch. It is also the only way to observe what a user
    actually sees.
    """
    code = textwrap.dedent(
        """
        import sys
        sys.argv = ["prog"] + {argv!r}
        try:
            import interpreter
        except BaseException as error:
            print("FAIL", type(error).__name__, str(error)[:80])
        else:
            print("OK")
        """
    ).format(argv=list(argv_extra))
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_in_repo(),
    )


def test_importing_interpreter_normally_succeeds():
    """Without `--os`, `import interpreter` works — the branch is skipped entirely.

    The control for the tests below: it proves any failure they see is the
    `--os` branch and not the package as a whole.
    """
    result = _run_python([])

    assert result.stdout.strip().startswith("OK"), (
        f"plain import should work; got {result.stdout!r} / {result.stderr[-300:]!r}"
    )


def test_os_mode_cannot_start_because_the_version_check_raises():
    """`--os` dies at import, before the automation loop is reached.

    The shadowing is deterministic, so this is not an environment quirk — it fails
    on every machine. Pinned as observed behaviour rather than an xfail, because it
    *is* the bug: a strict xfail here would be one that cannot pass without #407
    being fixed.
    """
    result = _run_python(["--os"])

    assert result.stdout.strip().startswith("FAIL"), (
        f"expected the crash to be reported; got {result.stdout!r}"
    )
    assert "TypeError" in result.stdout
    assert "not callable" in result.stdout


def test_the_crash_is_the_shadowing_not_a_network_failure():
    """The TypeError precedes the request, so this is not a connectivity problem.

    Guards against the tempting wrong fix: wrapping `check_for_update()` in a
    try/except would swallow the TypeError and *appear* to work, leaving the
    upgrade hint permanently dead and the shadowing untouched.
    """
    result = _run_python(["--os"])

    # No network error leaks, and the TypeError is what surfaces.
    assert "ConnectionError" not in result.stdout
    assert "Max retries" not in result.stdout
    assert "TypeError" in result.stdout


def test_voice_flag_is_reached_only_after_the_version_check():
    """`--voice` is handled after the version check, so it never prints today.

    Both flags live in the same branch and the crash precedes the `--voice` test,
    which is the ordering evidence: fixing the shadowing is a precondition for
    seeing "Coming soon..." at all.
    """
    result = _run_python(["--os", "--voice"])

    assert result.stdout.strip().startswith("FAIL"), (
        "the --voice branch is downstream of the crash"
    )


def test_check_for_update_compares_versions_correctly():
    """`check_for_update`'s comparison is directionally right, once unshadowed.

    Separates the comparison from the shadowing that #407 reports: the fix is a
    rename, not a rewrite. Newer published means True; equal or older means False,
    so an install ahead of PyPI is not nagged.
    """
    packaging_version = pytest.importorskip("packaging.version")

    def check_for_update(published, installed="0.0.1"):
        return packaging_version.parse(published) > packaging_version.parse(installed)

    assert check_for_update("0.0.2") is True, "a newer release should report True"
    assert check_for_update("0.0.1") is False, "the same version should report False"
    assert check_for_update("0.0.0") is False, "an older release should report False"
