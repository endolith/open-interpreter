"""Tests for ``interpreter.core.computer.ai.ai.fast_llm``.

``fast_llm`` temporarily swaps the interpreter's messages/system message, calls
the LLM, and restores the prior state. These tests pin both the happy path and
the failure path: an exception from the LLM must propagate (not be replaced by
an ``UnboundLocalError``) and the state must be restored either way.
"""

from types import SimpleNamespace

import pytest

from interpreter.core.computer.ai.ai import fast_llm


def _interpreter(chat):
    return SimpleNamespace(messages=["old"], system_message="old_sys", chat=chat)


def test_fast_llm_returns_content_and_restores_state():
    """fast_llm() returns the last message's content and restores the
    interpreter's messages/system message afterwards."""
    interpreter = _interpreter(lambda message: [{"content": "answer"}])
    llm = SimpleNamespace(interpreter=interpreter)

    result = fast_llm(llm, "sys", "user")

    assert result == "answer"
    assert interpreter.messages == ["old"]
    assert interpreter.system_message == "old_sys"


def test_fast_llm_propagates_chat_exception_and_restores_state():
    """If chat() raises, fast_llm() re-raises that original exception (rather
    than an UnboundLocalError from the finally block) and still restores the
    interpreter's messages/system message."""
    def chat(message):
        raise RuntimeError("boom")

    interpreter = _interpreter(chat)
    llm = SimpleNamespace(interpreter=interpreter)

    with pytest.raises(RuntimeError, match="boom"):
        fast_llm(llm, "sys", "user")

    assert interpreter.messages == ["old"]
    assert interpreter.system_message == "old_sys"
