import os
import sys

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


def patch_expanduser(monkeypatch, module, home):
    """Make expanduser('~') resolve to home (HOME is unreliable on Windows)."""

    monkeypatch.setattr(
        module.os.path,
        "expanduser",
        lambda path: str(home) if path == "~" else path,
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires an LLM API key (not run in default CI)"
    )


def pytest_collection_modifyitems(config, items):
    # Cloud agent runtime secrets are injected as normal environment variables
    # (e.g. OPENAI_API_KEY), so this check works the same locally and in CI.
    if os.environ.get("OPENAI_API_KEY"):
        return

    skip_integration = pytest.mark.skip(
        reason="OPENAI_API_KEY not set; skipping integration test"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
