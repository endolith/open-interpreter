"""Shared constants and helpers for unit tests.

Import from here (``from tests.helpers import ...``), not from conftest —
pytest loads conftest for hooks/fixtures but it is not a stable import path
on all platforms (notably Windows).
"""

import os
import platform

import pytest

# Model id used when a test needs an LLM name for branching (e.g. OpenAI key
# prompts). Matches tests/config.test.yaml; never hits a real API in unit tests.
TEST_LLM_MODEL = "gpt-4o-mini"

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


def patch_expanduser(monkeypatch, module, home):
    """Make expanduser('~') resolve to home (HOME is unreliable on Windows)."""

    monkeypatch.setattr(
        module.os.path,
        "expanduser",
        lambda path: str(home) if path == "~" else path,
    )
