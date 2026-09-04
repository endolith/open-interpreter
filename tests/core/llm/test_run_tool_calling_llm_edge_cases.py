"""Tests for run_tool_calling_llm edge cases not covered by existing tests."""

from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.llm.run_tool_calling_llm import process_messages, run_tool_calling_llm


def _make_llm():
    llm = mock.MagicMock()
    llm.model = "gpt-4o"
    llm.supports_functions = True
    return llm


def test_tool_calls_function_is_none_raises():
    """A tool_call delta with function=None crashes merge_deltas.

    KNOWN BUG: run_tool_calling_llm intends to skip tool_call deltas whose
    function attribute is None (the guard only converts deltas with a truthy
    function), but the unconverted delta is still forwarded to merge_deltas,
    which calls dict() on the tool_calls list and raises TypeError. Documenting
    current behavior; the fix belongs in a bug-fix PR.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {"delta": {"tool_calls": [SimpleNamespace(function=None)]}}
            ]
        }

    llm.completions = chat

    with pytest.raises(TypeError):
        list(run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "hi"}]}))


def test_tool_calls_empty_list():
    """An empty tool_calls list is handled without error.

    Streaming providers may emit a delta with an empty tool_calls list (e.g. a
    heartbeat chunk); run_tool_calling_llm must tolerate it instead of crashing,
    so a stray empty list never breaks a completion.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {"choices": [{"delta": {"tool_calls": []}}]}

    llm.completions = chat

    chunks = list(run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "hi"}]}))
    assert chunks == []


def test_process_messages_orphaned_function_response():
    """Orphaned function responses get a synthetic tool call prepended.

    When a conversation starts with a function result (e.g. history was
    truncated), the message must be paired with a synthetic assistant tool
    call so the LLM sees a well-formed tool interaction rather than an
    unanchored result.
    """
    messages = [
        {"role": "function", "name": "execute", "content": "output"},
    ]
    result = process_messages(messages)

    assert len(result) == 2
    assert result[0]["role"] == "assistant"
    assert result[0]["tool_calls"][0]["function"]["name"] == "execute"
    assert result[1]["role"] == "tool"
    assert result[1]["tool_call_id"] == result[0]["tool_calls"][0]["id"]


def test_process_messages_function_call_without_response():
    """Function call without a following function response gets an empty tool response.

    An assistant message that made a tool call must be followed by a tool
    response; supplying an empty one keeps the conversation well-formed for
    providers that reject an unpaired tool call.
    """
    messages = [
        {
            "role": "assistant",
            "function_call": {"name": "execute", "arguments": "{}"},
        },
    ]
    result = process_messages(messages)

    assert len(result) == 2
    assert result[0]["role"] == "assistant"
    assert result[0]["tool_calls"][0]["function"]["name"] == "execute"
    assert result[1]["role"] == "tool"
    assert result[1]["content"] == ""
    assert result[1]["tool_call_id"] == result[0]["tool_calls"][0]["id"]


def test_process_messages_preserves_non_tool_messages():
    """Non-tool messages are passed through unchanged.

    Ordinary user/assistant turns must not be altered by the tool-call
    normalization, otherwise conversation history would be corrupted before it
    reaches the model.
    """
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result = process_messages(messages)

    assert result == messages