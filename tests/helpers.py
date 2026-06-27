"""Shared constants and helpers for unit tests.

Import from here (``from tests.helpers import ...``), not from conftest —
pytest loads conftest for hooks/fixtures but it is not a stable import path
on all platforms (notably Windows).
"""

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


def patch_expanduser(monkeypatch, module, home):
    """Make expanduser('~') resolve to home (HOME is unreliable on Windows)."""

    monkeypatch.setattr(
        module.os.path,
        "expanduser",
        lambda path: str(home) if path == "~" else path,
    )
