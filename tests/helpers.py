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


# Subsystems returned by Computer._get_all_computer_tools_list (order matters).
COMPUTER_TOOL_SUBSYSTEMS = (
    "mouse",
    "keyboard",
    "display",
    "clipboard",
    "mail",
    "sms",
    "calendar",
    "contacts",
    "browser",
    "os",
    "vision",
    "skills",
    "docs",
    "ai",
    "files",
)


_BASH_COMPATIBLE_SHELL_NAMES = frozenset(
    {"bash", "sh", "dash", "zsh", "ksh", "ash"}
)


def require_bash_compatible_shell():
    """Skip when $SHELL is fish (or other non-bash) on Unix integration tests.

    Shell language execution now always invokes bash, but some tests still
    document the old $SHELL mismatch (#91) or run bash-syntax snippets directly.
    """
    if platform.system() == "Windows":
        return
    shell = os.environ.get("SHELL", "bash")
    shell_name = os.path.basename(shell).lower()
    if shell_name not in _BASH_COMPATIBLE_SHELL_NAMES:
        pytest.skip(
            f"SHELL={shell!r} is not bash-compatible. Export SHELL=/bin/bash for "
            f"integration tests, or rely on Shell using bash directly (see #91)."
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
    """Run nested bash loops through computer.run; fail fast on non-bash $SHELL."""

    require_bash_compatible_shell()
    chunks = list(
        interpreter.computer.run("shell", BASH_NESTED_LOOP_QUOTING_SNIPPET)
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
    """Make ``interpreter.core.computer.display.point.point`` importable in CI.

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
