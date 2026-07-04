import os
import platform

import pytest

_INTEGRATION_SKIP_REASON = (
    "integration tests need OI_RUN_INTEGRATION=1 (LLM auto-runs generated code)"
)


def _integration_tests_allowed() -> bool:
    # Local runs need an explicit opt-in even when OPENAI_API_KEY is set.
    # CI sets OI_RUN_INTEGRATION=1 in the integration workflow job.
    return os.environ.get("OI_RUN_INTEGRATION") == "1"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: calls an LLM and may execute generated code; "
        "set OI_RUN_INTEGRATION=1 locally",
    )


def pytest_collection_modifyitems(config, items):
    if not _integration_tests_allowed():
        skip_integration = pytest.mark.skip(reason=_INTEGRATION_SKIP_REASON)
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    _PLATFORM_MARKERS = {
        "linux_ci": "Linux",
        "windows_ci": "Windows",
        "darwin_ci": "Darwin",
    }
    current = platform.system()
    for item in items:
        for marker, required_os in _PLATFORM_MARKERS.items():
            if marker in item.keywords and current != required_os:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{marker} only runs on {required_os} (this host is {current})"
                    )
                )
