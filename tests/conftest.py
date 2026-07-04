import os

import pytest

_INTEGRATION_OPT_IN_SKIP = (
    "integration tests need OI_RUN_INTEGRATION=1 (LLM auto-runs generated code)"
)
_INTEGRATION_API_KEY_SKIP = "OPENAI_API_KEY not set; skipping integration tests"


def integration_skip_reason() -> str | None:
    """Return a pytest skip reason when integration tests should not run, else None."""
    if os.environ.get("OI_RUN_INTEGRATION") != "1":
        return _INTEGRATION_OPT_IN_SKIP
    if not os.environ.get("OPENAI_API_KEY"):
        return _INTEGRATION_API_KEY_SKIP
    return None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: calls an LLM and may execute generated code; "
        "requires OI_RUN_INTEGRATION=1 and OPENAI_API_KEY",
    )


def pytest_collection_modifyitems(config, items):
    reason = integration_skip_reason()
    if reason:
        skip_integration = pytest.mark.skip(reason=reason)
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
