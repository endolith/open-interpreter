import io
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires an LLM API key (not run in default CI)"
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("OPENAI_API_KEY"):
        return

    skip_integration = pytest.mark.skip(
        reason="OPENAI_API_KEY not set; skipping integration test"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


class FakeStdin(io.StringIO):
    """A stdin that reports whatever TTY status the test asks for."""

    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


@pytest.fixture
def tty(monkeypatch):
    """Run the test as if a person were at a terminal.

    prompt_choice refuses to read input when stdin is not a TTY, and pytest's
    stdin never is. A test that patches builtins.input to model someone typing
    an answer needs this too, or the prompt raises before input() is reached.
    """
    monkeypatch.setattr("sys.stdin", FakeStdin(tty=True))


@pytest.fixture
def no_tty(monkeypatch):
    """Run the test headless: stdin exists but is not a terminal."""
    monkeypatch.setattr("sys.stdin", FakeStdin(tty=False))
