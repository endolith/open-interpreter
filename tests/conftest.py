import os

import pytest


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
