import os
import platform

import pytest

from tests.helpers import INTEGRATION_SKIP_REASON
from tests.integration_support import (
    integration_tests_allowed,
    install_chat_approval,
)


def pytest_addoption(parser):
    parser.addoption(
        "--approve-integration",
        action="store_true",
        default=False,
        help="Auto-approve LLM-generated code in integration tests "
        "(sets OI_AUTO_APPROVE_INTEGRATION=1).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: calls an LLM and may execute generated code; "
        "requires OPENAI_API_KEY and OI_RUN_INTEGRATION=1 locally",
    )
    if config.getoption("--approve-integration"):
        os.environ["OI_AUTO_APPROVE_INTEGRATION"] = "1"


def pytest_sessionstart(session):
    if not integration_tests_allowed():
        return
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return
    if os.environ.get("OI_AUTO_APPROVE_INTEGRATION") == "1":
        print(
            "\nIntegration tests: LLM-generated code will run without per-block prompts "
            "(OI_AUTO_APPROVE_INTEGRATION=1).\n"
        )
        return
    print(
        "\nIntegration tests call an LLM and execute generated code on your machine. "
        "You will be prompted before each code block (y/n/a).\n"
    )


def pytest_collection_modifyitems(config, items):
    if not integration_tests_allowed():
        skip_integration = pytest.mark.skip(reason=INTEGRATION_SKIP_REASON)
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


@pytest.fixture(autouse=True)
def approve_llm_generated_code(request, monkeypatch):
    if "integration" not in request.node.keywords:
        return
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return
    install_chat_approval(monkeypatch, request.node.name)
