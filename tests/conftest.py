import os
import platform

import pytest

_INTEGRATION_OPT_IN_SKIP = (
    "integration tests need OI_RUN_INTEGRATION=1 (LLM auto-runs generated code)"
)
_INTEGRATION_API_KEY_SKIP = "OPENAI_API_KEY not set; skipping integration tests"


def _conversations_dir() -> str:
    """Real conversations log dir, resolved lazily to avoid import cost."""
    import platformdirs

    return os.path.join(platformdirs.user_config_dir("open-interpreter"), "conversations")


_test_created_files: list[str] = []


def _install_creation_audit_hook() -> None:
    """Log files the test process itself creates under the conversations dir.

    sys.addaudithook fires on every open() process-wide; the hook records
    only paths under the conversations folder opened for writing that did
    not exist yet. Files created by other processes (e.g. the user chatting
    mid-run) never trigger our hook, so they can never be mistaken for test
    artifacts — unlike a before/after directory snapshot.
    """
    import sys

    prefix = _conversations_dir() + os.sep
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND

    def _hook(event: str, args: tuple) -> None:
        if event != "open":
            return
        try:
            path, _, flags = args[0], args[1], args[2]
        except (IndexError, TypeError):
            return
        if not isinstance(path, str) or not path.startswith(prefix):
            return
        if not flags & write_flags:
            return
        try:
            exists = os.path.exists(path)
        except OSError:
            return
        if not exists:
            _test_created_files.append(path)

    sys.addaudithook(_hook)


_install_creation_audit_hook()


@pytest.fixture(autouse=True)
def _clean_test_conversations():
    """Delete conversation logs the test run writes to the real log folder.

    Tests exercise the real save path (history stays enabled), so chat()
    writes JSON logs into the user's conversations folder. The audit hook
    above records exactly the files this pytest process created, and only
    those are removed afterwards. Behaves identically on CI and locally.
    """
    del _test_created_files[:]
    yield
    while _test_created_files:
        try:
            os.remove(_test_created_files.pop())
        except OSError:
            pass


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
    config.addinivalue_line(
        "markers",
        "mock_llm: uses a local OpenAI-compatible HTTP server (no API key)",
    )


def pytest_collection_modifyitems(config, items):
    reason = integration_skip_reason()
    if reason:
        skip_integration = pytest.mark.skip(reason=reason)
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
