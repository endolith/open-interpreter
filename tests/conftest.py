import functools
import os
import platform

import litellm
import pytest

_INTEGRATION_OPT_IN_SKIP = (
    "integration tests need OI_RUN_INTEGRATION=1 (LLM auto-runs generated code)"
)
_INTEGRATION_API_KEY_SKIP = "OPENAI_API_KEY not set; skipping integration tests"


from tests.helpers import (
    begin_test_conversations,
    end_test_conversations,
    _test_interpreters,
)


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
    reported log path is removed unless that file already existed before
    the test ran. Identity comes from the writer object itself rather than
    filesystem observation, so files from anyone else are never touched.
    Cleanup stays inside the snapshotted directory, and a snapshot that
    fails to read removes nothing. Behaves identically on CI and locally.
    """
    state = begin_test_conversations()
    yield
    end_test_conversations(state)


_LOCAL_MODEL_PREFIXES = ("ollama/", "lm_studio/", "lm-studio/")


def integration_skip_reason() -> str | None:
    """Return a pytest skip reason when integration tests should not run, else None.

    The required key comes from the model under test, not a fixed provider:
    litellm knows which env var each provider needs, so `groq/...` checks
    GROQ_API_KEY and `claude...` checks ANTHROPIC_API_KEY. Local providers
    need no key, and an unset OI_INTEGRATION_MODEL keeps the OpenAI default.
    """
    if os.environ.get("OI_RUN_INTEGRATION") != "1":
        return _INTEGRATION_OPT_IN_SKIP
    model = os.environ.get("OI_INTEGRATION_MODEL", "")
    if model.startswith(_LOCAL_MODEL_PREFIXES):
        return None
    if not model:
        if not os.environ.get("OPENAI_API_KEY"):
            return _INTEGRATION_API_KEY_SKIP
        return None
    try:
        environment = litellm.validate_environment(model)
    except Exception:
        # Let the real call surface an unfamiliar provider's error.
        return None
    if not environment.get("keys_in_environment", True):
        missing = ", ".join(environment.get("missing_keys") or ["provider credentials"])
        return f"{missing} not set; skipping integration tests"
    return None


def pytest_collection_modifyitems(config, items):
    """Skip integration tests without opt-in/credentials, and wrong-OS markers."""
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
