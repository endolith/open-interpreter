"""Shared constants and helpers for unit tests.

Import from here (``from tests.helpers import ...``), not from conftest —
pytest loads conftest for hooks/fixtures but it is not a stable import path
on all platforms (notably Windows).
"""

import os
import platform
import shutil

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


_BASH_COMPATIBLE_SHELL_NAMES = frozenset(
    {"bash", "sh", "dash", "zsh", "ksh", "ash"}
)


def require_bash_compatible_shell():
    """Fail immediately if Shell would spawn a non-bash-compatible $SHELL.

    OI feeds bash-syntax snippets to subprocess_language, which uses
    os.environ["SHELL"] on Unix. Fish and other shells hang waiting for
    ##end_of_execution## instead of erroring.
    """
    if platform.system() == "Windows":
        return
    shell = os.environ.get("SHELL", "bash")
    shell_name = os.path.basename(shell).lower()
    if shell_name not in _BASH_COMPATIBLE_SHELL_NAMES:
        pytest.fail(
            f"SHELL={shell!r} cannot run bash-syntax shell code (Shell uses "
            f"os.environ['SHELL']). Use bash or wait for explicit bash in develop."
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
