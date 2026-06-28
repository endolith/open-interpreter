import os
import platform

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: calls an LLM and may execute generated code; requires OPENAI_API_KEY",
    )


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("OPENAI_API_KEY"):
        skip_integration = pytest.mark.skip(
            reason="integration tests need OPENAI_API_KEY (LLM may run generated code)"
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
