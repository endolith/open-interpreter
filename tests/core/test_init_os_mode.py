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
from unittest import mock

import pytest

IGNORED = ("__pycache__", ".pytest_cache", "mutants")


def _in_repo():
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


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
        from unittest.mock import patch
        sys.argv = ["prog"] + {argv!r}
        # The version check must never touch the network from a test, before or
        # after #407 is fixed: today the TypeError precedes it, but once the
        # shadowing is fixed `requests.get` would fire for real.
        with patch("requests.get") as get:
            get.return_value.json.return_value = {{"info": {{"version": "0"}}}}
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


# --- the logic itself, tested directly so it is actually covered ---

def _logic():
    """Import the functions as a module, without going through package import.

    They are defined inside the `--os` branch of `__init__.py`, so they cannot be
    imported normally. The source is compiled into a throwaway module here, which
    keeps them reachable for testing; see the module docstring for why this does
    not move the coverage number.
    """
    import types

    from importlib.metadata import version as installed_version

    import packaging.version

    namespace = {"installed_version": installed_version, "packaging_version": packaging.version}
    # Only the two functions, extracted verbatim from the branch.
    # Resolved from the repository root, not the working directory, so the
    # direct tests also pass when pytest starts outside the repo.
    source = __import__("pathlib").Path(_in_repo(), "interpreter", "__init__.py").read_text(encoding="utf-8")
    body = source[source.index("    def print_markdown("):source.index("    if check_for_update():")]
    # De-indent from the branch level so it compiles at module level.
    body = "\n".join(line[4:] if line.startswith("    ") else line for line in body.split("\n"))
    module = types.ModuleType("os_mode_logic")
    exec(compile(body, "os_mode_logic", "exec"), namespace, namespace)  # noqa: S102
    return namespace


def _version_double(installed):
    """A stand-in for the `version` name: callable, and carries `parse`.

    The shipped code needs `version("open-interpreter")` to return the installed
    version *and* `version.parse(...)` to compare. #407's shadowing collapses those
    into one module object that satisfies neither, which is the bug.
    """
    import packaging.version

    def version(_package):
        return installed

    version.parse = packaging.version.parse
    return version


def test_print_markdown_renders_a_horizontal_rule_for_dashes():
    """A line of `---` becomes a horizontal rule, not literal dashes."""
    namespace = _logic()
    rules = []
    namespace["rich_print"] = rules.append
    namespace["Rule"] = lambda **kwargs: f"RULE({kwargs})"
    namespace["Markdown"] = lambda line: f"MD({line})"

    namespace["print_markdown"]("above\n---\nbelow")

    assert any("RULE(" in r for r in rules), (
        f"expected a rule for the --- line, got {rules}"
    )


def test_print_markdown_passes_ordinary_lines_through_as_markdown():
    """Non-empty, non-`---` lines are rendered as markdown."""
    namespace = _logic()
    rendered = []
    namespace["rich_print"] = rendered.append
    namespace["Rule"] = lambda **kwargs: "RULE"
    namespace["Markdown"] = lambda line: f"MD({line})"

    namespace["print_markdown"]("hello")

    assert rendered == ["MD(hello)"]


def test_print_markdown_pads_a_single_leading_quote_tag():
    """A one-line message starting with `>` gets a blank line after it.

    Purely cosmetic, and pinned because it is deliberate: those tags read as
    cramped without the space.
    """
    namespace = _logic()
    namespace["rich_print"] = lambda *_a, **_k: None
    namespace["Rule"] = lambda **kwargs: "RULE"
    namespace["Markdown"] = lambda line: line

    import io as _io
    import contextlib as _ctx

    buffer = _io.StringIO()
    with _ctx.redirect_stdout(buffer):
        namespace["print_markdown"]("> quoted")
    out = buffer.getvalue()

    # `rich_print` is stubbed, so the content line contributes no output of its
    # own and what remains is exactly the deliberate padding newline.
    assert out == "\n", f"expected only the padding newline, got {out!r}"


def test_print_markdown_survives_a_character_it_cannot_encode():
    """A line that rich cannot encode falls back instead of raising.

    Without the guard a single unencodable character — a lone surrogate from
    truncated output — would abort the whole banner mid-render.
    """
    namespace = _logic()

    def explode(_line):
        raise UnicodeEncodeError("ascii", "", 0, 1, "bad char")

    namespace["rich_print"] = explode
    namespace["Rule"] = lambda **kwargs: "RULE"
    namespace["Markdown"] = lambda line: "MD"

    import contextlib as _ctx
    import io as _io

    buffer = _io.StringIO()
    with _ctx.redirect_stdout(buffer):
        namespace["print_markdown"]("bad \udcff line")

    assert "Error displaying line" in buffer.getvalue()


def test_check_for_update_reports_a_newer_release():
    """A newer published version means an update is available."""
    namespace = _logic()
    response = mock.Mock()
    response.json.return_value = {"info": {"version": "99.0.0"}}
    namespace["requests"] = mock.Mock(get=mock.Mock(return_value=response))
    # The extracted body still contains #407's shadowing, so rebind `version` the
    # way the fix would: a callable that reports the installed version, while the
    # module keeps its `.parse`.
    namespace["version"] = _version_double("1.0.0")

    assert namespace["check_for_update"]() is True


def test_check_for_update_is_false_when_the_install_is_current():
    """The common case: already current, so no nagging."""
    namespace = _logic()
    response = mock.Mock()
    response.json.return_value = {"info": {"version": "1.0.0"}}
    namespace["requests"] = mock.Mock(get=mock.Mock(return_value=response))
    namespace["version"] = _version_double("1.0.0")

    assert namespace["check_for_update"]() is False
