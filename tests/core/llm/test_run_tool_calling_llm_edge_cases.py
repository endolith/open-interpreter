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


def test_tool_calls_function_is_none_skipped():
    """A tool_call delta with function=None yields no chunks.

    Deltas whose function attribute is None carry nothing mergeable, so the
    tool_calls key is dropped and the chunk passes through empty instead of
    crashing merge_deltas. Regression test for #254.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {"choices": [{"delta": {"tool_calls": [SimpleNamespace(function=None)]}}]}

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "hi"}]})) == []


def test_tool_calls_function_is_none_no_auth_error(monkeypatch):
    """A function-less delta does not trip the auth judge-layer guard.

    The detection flag is only set when a valid function call is converted,
    so with INTERPRETER_REQUIRE_AUTHENTICATION enabled a malformed delta
    yields no chunks instead of raising "Judge layer required but did not
    run.". Regression test for #254.
    """
    monkeypatch.setenv("INTERPRETER_REQUIRE_AUTHENTICATION", "true")
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {"choices": [{"delta": {"tool_calls": [SimpleNamespace(function=None)]}}]}

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "hi"}]})) == []


def test_tool_calls_later_valid_entry_converted():
    """A valid entry after a function-less one is still converted.

    Only entry 0 used to be inspected, so a leading malformed entry hid a
    later valid tool call. Every entry is now scanned by index.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            SimpleNamespace(function=None),
                            SimpleNamespace(
                                id="toolu_1",
                                function=SimpleNamespace(
                                    name="execute",
                                    arguments='{"language": "python", "code": "print(1)"}',
                                ),
                            ),
                        ]
                    }
                }
            ]
        }

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"}
    ]


def test_tool_calls_dict_shaped_entries():
    """Dict-shaped entries (as newer litellm versions emit) are handled.

    Entries may be plain dicts instead of objects; a dict entry with a valid
    function mapping converts the same way.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"function": None},
                            {
                                "id": "toolu_1",
                                "function": {
                                    "name": "execute",
                                    "arguments": '{"language": "python", "code": "print(1)"}',
                                },
                            },
                        ]
                    }
                }
            ]
        }

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"}
    ]


def test_tool_calls_nameless_continuation_merges():
    """A continuation chunk without a name still appends its arguments.

    Streaming providers send the tool name once, then arguments-only deltas.
    The nameless chunk merges into the accumulated call so the full code is
    yielded; the missing name must not be treated as a malformed entry.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            SimpleNamespace(
                                id="toolu_1",
                                function=SimpleNamespace(
                                    name="execute",
                                    arguments='{"language": "python", "code": "pri',
                                ),
                            )
                        ]
                    }
                }
            ]
        }
        yield {"choices": [{"delta": {"tool_calls": [SimpleNamespace(function=SimpleNamespace(arguments='nt(1)"}'))]}}]}

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "pri"},
        {"type": "code", "format": "python", "content": "nt(1)"},
    ]


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