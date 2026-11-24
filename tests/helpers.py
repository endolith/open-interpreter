"""Shared constants and helpers for unit tests.

Import from here (``from tests.helpers import ...``), not from conftest —
pytest loads conftest for hooks/fixtures but it is not a stable import path
on all platforms (notably Windows).
"""

import os
import platform
import shutil
import sys
import types

import pytest

# Model id used when a test needs an LLM name for branching (e.g. OpenAI key
# prompts). Matches tests/config.test.yaml; never hits a real API in unit tests.
TEST_LLM_MODEL = "gpt-4o-mini"


def chunks_of_type(chunks, chunk_type):
    return [chunk for chunk in chunks if chunk.get("type") == chunk_type]


def require_chrome_for_html():
    """Skip when no Chrome/Chromium binary is available for html2image."""

    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if shutil.which(name):
            return
    pytest.skip("google-chrome or chromium not installed (needed for HTML/React e2e)")


def _run_chrome_screenshot(executable, html, headless, flags, output_dir, size, timeout):
    """Render ``html`` with one executable; return None on success, else why not.

    Mirrors html2image's ``ChromiumHeadless.screenshot`` command and captures
    the output the product discards.
    """
    import subprocess
    import tempfile
    import uuid

    output_path = os.path.join(output_dir, f"chrome-probe-{uuid.uuid4().hex}.png")
    workdir = tempfile.mkdtemp(prefix="oi-chrome-probe-")
    input_path = os.path.join(workdir, "probe.html")
    try:
        with open(input_path, "w", encoding="utf-8") as file:
            file.write(html)
        command = [
            executable,
            headless,
            f"--screenshot={output_path}",
            f"--window-size={size[0]},{size[1]}",
            *flags,
            input_path,
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout
            )
        except FileNotFoundError:
            return f"{executable} is not runnable"
        except subprocess.TimeoutExpired:
            return f"{executable} timed out after {timeout}s"

        try:
            size_bytes = os.path.getsize(output_path)
        except OSError:
            size_bytes = 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            os.remove(output_path)
        except OSError:
            pass

    if completed.returncode == 0 and size_bytes > 0:
        return None
    output = (completed.stderr or completed.stdout).strip()
    parts = [
        f"{executable} exited {completed.returncode}"
        + (" (signal)" if completed.returncode < 0 else ""),
        "wrote no file" if not size_bytes else f"wrote {size_bytes} bytes",
        f"output: {output[:250]}" if output else "no output",
    ]
    return ", ".join(parts)


def chrome_screenshot_diagnostic(html, size=(960, 540), timeout=90):
    """Return None when the html2image screenshot pipeline works, else why not.

    ``html_to_png_base64`` deliberately swallows every exception and runs the
    browser with stdout/stderr discarded, so a machine that cannot render
    reaches these tests as a missing image chunk with no explanation. This
    repeats the same invocation with the browser's output captured, so the
    callers can skip with the actual reason instead of asserting blindly.

    html2image resolves the browser itself (on Linux it prefers ``chromium``
    over ``google-chrome``), so when that executable fails but another one is
    installed, both outcomes are reported: it distinguishes a broken default
    browser from a machine with no working browser at all. The returned string
    is one line, safe to put in a skip reason.
    """
    import html2image

    from interpreter.terminal_interface.utils.local_storage_path import (
        get_storage_path,
    )

    def summarize(*parts):
        text = " | ".join(str(part) for part in parts if part)
        return " ".join(text.split())[:500]

    try:
        browser = html2image.Html2Image(disable_logging=True).browser
    except Exception as exc:
        return summarize("html2image found no browser:", type(exc).__name__, exc)

    headless = "--headless"
    if browser.use_new_headless is not None:
        headless += "=new" if browser.use_new_headless else "=old"

    output_dir = get_storage_path()
    failure = _run_chrome_screenshot(
        browser.executable, html, headless, browser.flags, output_dir, size, timeout
    )
    if failure is None:
        return None

    details = [failure]
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if not path:
            continue
        if os.path.basename(browser.executable) == name or path == browser.executable:
            continue
        other = _run_chrome_screenshot(
            path, html, headless, browser.flags, output_dir, size, timeout
        )
        details.append(f"{name}: " + ("renders fine" if other is None else other))
    return summarize(*details)


_BASH_COMPATIBLE_SHELL_NAMES = frozenset(
    {"bash", "sh", "dash", "zsh", "ksh", "ash"}
)


def require_bash_compatible_shell():
    """Skip unless the test shell is bash-compatible.

    Shell-language execution now invokes bash directly, but some tests still
    feed bash-syntax snippets to the process directly. Skip rather than fail
    when that prerequisite is absent.
    """
    if platform.system() == "Windows":
        return
    shell = os.environ.get("SHELL", "bash")
    shell_name = os.path.basename(shell).lower()
    if shell_name not in _BASH_COMPATIBLE_SHELL_NAMES:
        pytest.skip(
            f"SHELL={shell!r} is not bash-compatible; set SHELL=/bin/bash to "
            "run bash-syntax tests."
        )


def console_output_text(chunks):
    """Join console output chunks from ``computer.run`` / ``terminal.run``."""

    return "".join(
        chunk.get("content", "")
        for chunk in chunks
        if chunk.get("format") == "output"
    )


# Bash nested-loop quoting smoke shared by linux_ci and darwin_ci jobs.
# Linux CI excludes darwin_ci markers; macOS CI runs only darwin_ci — so we
# keep thin per-runner tests that call this helper rather than one dual-marked test.
BASH_NESTED_LOOP_QUOTING_SNIPPET = (
    'for i in a b; do for j in 1 2; do echo "${i}_${j}"; done; done'
)


def assert_bash_nested_loop_output(output):
    assert "a_1" in output
    assert "b_2" in output


def run_bash_nested_loop_quoting_smoke(interpreter):
    """Run nested bash loops through computer.run; skip on non-bash $SHELL."""

    require_bash_compatible_shell()
    chunks = list(
        interpreter.toolbox.run("bash", BASH_NESTED_LOOP_QUOTING_SNIPPET)
    )
    assert_bash_nested_loop_output(console_output_text(chunks))


def patch_expanduser(monkeypatch, module, home):
    """Make expanduser('~') resolve to home (HOME is unreliable on Windows)."""

    monkeypatch.setattr(
        module.os.path,
        "expanduser",
        lambda path: str(home) if path == "~" else path,
    )


def install_point_heavy_deps(monkeypatch):
    """Make ``interpreter.core.toolbox.display.point.point`` importable in CI.

    ``point.py`` imports torch, sentence_transformers, timm, nltk and cv2
    unconditionally at module level. Those are [computer]-optional deps that are
    never installed in unit-test CI (and their absence crashes ``lazy_import``),
    so tests that only exercise ``point``'s dispatch/geometry logic install bare
    stub modules for them before the module is first imported. The stubs are
    removed again by pytest's monkeypatch teardown.
    """

    from types import SimpleNamespace

    def _mod(name):
        return types.ModuleType(name)

    nltk_words = _mod("nltk.corpus.words")
    nltk_words.words = lambda: []
    nltk_corpus = _mod("nltk.corpus")
    nltk_corpus.words = nltk_words
    nltk = _mod("nltk")
    nltk.corpus = nltk_corpus
    nltk.download = lambda *a, **k: None

    torch = _mod("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    torch.device = lambda name: "cpu"
    torch.stack = lambda *a, **k: None
    torch.cat = lambda *a, **k: None

    class _StubSentenceTransformer:
        def __init__(self, *a, **k):
            pass

        def to(self, device):
            return self

        def encode(self, *a, **k):
            return []

    sentence_transformers = _mod("sentence_transformers")
    sentence_transformers.SentenceTransformer = _StubSentenceTransformer
    sentence_transformers.util = SimpleNamespace(
        semantic_search=lambda *a, **k: []
    )

    timm = _mod("timm")
    timm.create_model = lambda *a, **k: None
    timm.data = SimpleNamespace(
        resolve_model_data_config=lambda *a, **k: {},
        create_transform=lambda *a, **k: None,
    )

    cv2 = _mod("cv2")

    for name, module in (
        ("nltk", nltk),
        ("nltk.corpus", nltk_corpus),
        ("nltk.corpus.words", nltk_words),
        ("torch", torch),
        ("sentence_transformers", sentence_transformers),
        ("timm", timm),
        ("cv2", cv2),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _conversations_dir() -> str:
    """Real conversations log dir, resolved lazily to avoid import cost."""
    import platformdirs

    return os.path.join(platformdirs.user_config_dir("open-interpreter"), "conversations")


# Interpreter instances tracked for conversation cleanup. Teardown retains
# pre-existing registrations and drops entries added during the active cycle.
_test_interpreters: list = []


def _conversation_file(interpreter) -> str | None:
    """Reported conversation log path, or None.

    chat() writes history_path/conversation_filename synchronously when it
    saves. A configured filename can also name a pre-existing or
    never-created path — reporting is not proof of a save, which is why
    teardown additionally requires the path to be absent from the
    pre-existing snapshot. The candidate must stay inside the resolved
    history directory:
    os.path.join() returns an absolute filename unchanged, so names like
    ../user.json or /etc/passwd.json would otherwise escape. Returns None
    for anything outside, for non-.json names (production saves always end
    in .json), and when history is off or no filename is set.
    """
    if not interpreter.conversation_history or not interpreter.conversation_filename:
        return None
    try:
        history_dir = os.path.realpath(interpreter.conversation_history_path)
        candidate = os.path.realpath(
            os.path.join(history_dir, interpreter.conversation_filename)
        )
    except (OSError, ValueError, TypeError):
        return None
    try:
        if os.path.commonpath((history_dir, candidate)) != history_dir:
            return None
    except ValueError:
        return None
    if not candidate.endswith(".json"):
        return None
    return candidate


def _snapshot_dir(path: str) -> frozenset | None:
    """Canonical paths present in a directory now, or None on failure.

    A failed listing (missing directory, permission error) returns None —
    never an empty set — so callers cannot mistake "could not read" for
    "nothing was there" and delete a pre-existing file on that basis.
    """
    try:
        entries = os.listdir(path)
    except OSError:
        return None
    return frozenset(os.path.realpath(os.path.join(path, name)) for name in entries)


def _is_within(directory: str, path: str) -> bool:
    """Whether a canonical path stays inside a canonical directory."""
    try:
        return os.path.commonpath((directory, path)) == directory
    except ValueError:
        return False


def _remove_test_conversations(
    interpreters, pre_existing=frozenset(), history_dir=None
) -> None:
    """Remove log files reported by the given interpreters.

    Only reported paths absent from pre_existing (a snapshot of the log dir
    taken before the test ran) are removed: a reported path for a file that
    already existed means the test never saved it — the filename was merely
    configured — so it is preserved. When history_dir is given, removal is
    additionally restricted to paths inside it, so interpreters configured
    with custom directories can never cause deletion elsewhere. A failed
    snapshot (pre_existing None) removes nothing. Missing files and removal
    errors are tolerated.
    """
    if pre_existing is None:
        return
    for interpreter in interpreters:
        try:
            path = _conversation_file(interpreter)
        except Exception:
            continue
        if path is None or path in pre_existing:
            continue
        if history_dir is not None and not _is_within(history_dir, path):
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def begin_test_conversations():
    """Snapshot state for one fixture cycle; return a token for the matching teardown.

    Records the canonical history dir, its pre-existing snapshot, and how
    many interpreters are already tracked. Instances registered before this
    call (e.g. module-level ones built at import time) count as
    pre-existing tracking, not per-test additions.
    """
    history_dir = os.path.realpath(_conversations_dir())
    return history_dir, _snapshot_dir(history_dir), len(_test_interpreters)


def end_test_conversations(state) -> None:
    """Run one teardown cycle, retaining previously tracked interpreters.

    Removes reported log files for all tracked instances (so a reused
    module-level interpreter is still cleaned on later cycles), then drops
    only the instances registered since the matching setup call. Pre-existing
    registrations survive for the next cycle instead of being forgotten after
    the first teardown.
    """
    history_dir, before, already_tracked = state
    try:
        _remove_test_conversations(_test_interpreters, before, history_dir)
    finally:
        del _test_interpreters[already_tracked:]
