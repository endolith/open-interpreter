"""Terminal-interface test fixtures."""

import os

import pytest


@pytest.fixture(autouse=True)
def _force_xterm_terminal():
    """Rich Live emits cursor-up/erase sequences only on non-dumb terminals."""
    previous = os.environ.get("TERM")
    os.environ["TERM"] = "xterm-256color"
    yield
    if previous is None:
        os.environ.pop("TERM", None)
    else:
        os.environ["TERM"] = previous
