import shutil

import pytest

from interpreter.core.terminal.languages.augeas import Augeas


def test_preprocess_keeps_the_command_and_adds_a_newline():
    """A block is passed to augtool verbatim with a trailing newline."""
    assert Augeas().preprocess_code("  get /augeas/version  ") == "get /augeas/version\n"


def test_detect_end_of_execution_is_never_prompt_based():
    """One-shot execution does not rely on augtool's tty-only prompt."""
    assert Augeas().detect_end_of_execution("augtool> ") is False


@pytest.mark.skipif(shutil.which("augtool") is None, reason="augtool not installed")
def test_run_returns_output_without_hanging():
    """A real block completes and returns augtool's output (regression: every run hung)."""
    chunks = list(Augeas().run("get /augeas/version"))
    text = "".join(chunk.get("content", "") for chunk in chunks)
    assert "/augeas/version =" in text


@pytest.mark.skipif(shutil.which("augtool") is None, reason="augtool not installed")
def test_run_reports_invalid_commands():
    """Invalid augtool input is reported instead of leaving the turn hanging."""
    chunks = list(Augeas().run("this is not an augtool command"))
    assert chunks
    assert "error" in "".join(c.get("content", "") for c in chunks).lower()
