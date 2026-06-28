import os
import platform

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires an LLM API key (not run in default CI)"
    )


def pytest_collection_modifyitems(config, items):
    # Cloud agent runtime secrets are injected as normal environment variables
    # (e.g. OPENAI_API_KEY), so this check works the same locally and in CI.
    if not os.environ.get("OPENAI_API_KEY"):
        skip_integration = pytest.mark.skip(
            reason="OPENAI_API_KEY not set; skipping integration test"
        )
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
