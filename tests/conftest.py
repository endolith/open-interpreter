import functools
import os
import platform

import pytest

_INTEGRATION_OPT_IN_SKIP = (
    "integration tests need OI_RUN_INTEGRATION=1 (LLM auto-runs generated code)"
)
_INTEGRATION_API_KEY_SKIP = "OPENAI_API_KEY not set; skipping integration tests"


def _conversation_file(interpreter) -> str | None:
    """Exact log path an interpreter instance saved to, or None.

    chat() writes history_path/conversation_filename synchronously before
    returning, so a filename set on the instance names a file this process
    wrote — no guessing, no timestamps, no other process can hold the same
    claim unless it wrote the identical path first (in which case our save
    already overwrote it during the test).
    """
    if not interpreter.conversation_history or not interpreter.conversation_filename:
        return None
    path = os.path.join(
        interpreter.conversation_history_path, interpreter.conversation_filename
    )
    if not path.endswith(".json"):
        return None
    return path


_test_interpreters: list = []


def _remove_test_conversations(interpreters) -> None:
    """Remove log files reported by the given interpreters, tolerating failures."""
    for interpreter in interpreters:
        try:
            path = _conversation_file(interpreter)
        except Exception:
            continue
        if path is None:
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def _install_construction_tracking() -> None:
    """Register every OpenInterpreter built in this process for teardown.

    Installed once at conftest import time — pytest always loads conftest
    before collecting any test module — so even module-level instances built
    at import time (e.g. tests/test_interpreter.py) are tracked. A per-test
    patch would miss those, leaving their conversation files behind whenever
    they chat (today that means integration runs). The wrapper only appends
    to a list; per-test teardown below drains it.
    """
    from interpreter.core.core import OpenInterpreter

    orig_init = OpenInterpreter.__init__

    @functools.wraps(orig_init)
    def tracking_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _test_interpreters.append(self)

    OpenInterpreter.__init__ = tracking_init


_install_construction_tracking()


@pytest.fixture(autouse=True)
def _clean_test_conversations():
    """Delete conversation logs tests write to the real log folder.

    Tests exercise the real save path (history stays enabled), so chat()
    writes JSON logs into the user's conversations folder. Every
    OpenInterpreter constructed in this process is registered (including
    module-level instances built at import time); at teardown each one's
    reported log path is removed. Identity comes from the writer object
    itself rather than filesystem observation, so files from anyone else
    are never touched. Behaves identically on CI and locally.
    """
    yield
    try:
        _remove_test_conversations(_test_interpreters)
    finally:
        del _test_interpreters[:]


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
