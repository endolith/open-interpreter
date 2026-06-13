import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires an LLM API key (not run in default CI)"
    )
    config.addinivalue_line(
        "markers", "mock_llm: uses local mock OpenAI-compatible HTTP server"
    )
    config.addinivalue_line(
        "markers",
        "openrouter: requires OPENROUTER_API_KEY (optional CI smoke test)",
    )


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("OPENAI_API_KEY"):
        skip_integration = pytest.mark.skip(
            reason="OPENAI_API_KEY not set; skipping integration test"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    if not os.environ.get("OPENROUTER_API_KEY"):
        skip_openrouter = pytest.mark.skip(
            reason="OPENROUTER_API_KEY not set; skipping openrouter smoke test"
        )
        for item in items:
            if "openrouter" in item.keywords:
                item.add_marker(skip_openrouter)
