"""Tests for run_tool_calling_llm edge cases not covered by existing tests."""

from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.llm.run_tool_calling_llm import process_messages, run_tool_calling_llm


def _make_llm():
    llm = mock.MagicMock()
    llm.model = "gpt-4o"
    llm.supports_functions = True
    # Seed an empty queue: without this, MagicMock auto-creates a truthy
    # _pending_tool_calls, the runner takes the replay branch, and these
    # tests never exercise the completion stream.
    llm.interpreter._pending_tool_calls = []
    return llm


def test_tool_calls_function_is_none_skipped():
    """A tool_call delta with function=None is skipped, not crashed on.

    #415 fixed the known bug where run_tool_calling_llm forwarded
    unconverted deltas to merge_deltas, which called dict() on the
    tool_calls list and raised TypeError. Now the delta is skipped.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {"delta": {"tool_calls": [SimpleNamespace(function=None)]}}
            ]
        }

    llm.completions = chat

    result = list(run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "hi"}]}))
    assert result == []


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

def test_dict_entries_with_none_index_stay_distinct():
    """Dict entries with "index": None fall back to id, not a shared key.

    Serialized non-streamed responses often carry "index": None. Keying them
    all as ("index", None) merges the second call's arguments into the first
    and both calls are lost; falling back to the id keeps them distinct.
    """
    from interpreter.core.llm.run_tool_calling_llm import _tool_call_entry_key

    first = _tool_call_entry_key({"index": None, "id": "a"}, 0)
    second = _tool_call_entry_key({"index": None, "id": "b"}, 1)
    assert first == "a"
    assert second == "b"
    assert first != second


def test_new_user_turn_clears_stale_queued_calls():
    """A new user message supersedes calls queued by an earlier response.

    Without this, declining a call (or any interruption) leaves entries in
    _pending_tool_calls, and the next chat() replays a stale call — executing
    code without spending an LLM request — instead of answering the user.
    """
    llm = _make_llm()
    llm.interpreter._pending_tool_calls = [
        {"id": "stale", "name": "execute", "arguments": "{}"}
    ]

    def chat(**params):
        yield {"choices": [{"delta": {"content": "fresh answer"}}]}

    llm.completions = chat
    llm.interpreter.messages = [{"role": "user", "content": "hi"}]

    chunks = list(
        run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "hi"}]})
    )
    assert llm.interpreter._pending_tool_calls == []
    assert chunks == [{"type": "message", "content": "fresh answer"}]


def test_replayed_call_carries_original_review_verdict(monkeypatch):
    """A queued call replays its response's judge verdict under auth mode.

    With INTERPRETER_REQUIRE_AUTHENTICATION=true the runner raises unless a
    function call is accompanied by review content. The replayed chunk has
    none, so the verdict stamped at queue time is restored — but only when
    the original response actually passed review.
    """
    monkeypatch.setenv("INTERPRETER_REQUIRE_AUTHENTICATION", "true")

    llm = _make_llm()
    llm.interpreter._pending_tool_calls = [
        {"id": "q", "name": "execute", "arguments": '{"language": "python", "code": "print(1)"}', "review": "looks safe"}
    ]
    chunks = list(
        run_tool_calling_llm(llm, {"messages": [{"role": "assistant", "content": ""}]})
    )
    assert llm.interpreter._pending_tool_calls == []
    assert any(c.get("type") == "code" for c in chunks)

    llm = _make_llm()
    llm.interpreter._pending_tool_calls = [
        {"id": "q", "name": "execute", "arguments": '{\"language\": \"python\", \"code\": \"print(1)\"}'}
    ]
    with pytest.raises(Exception, match="Judge layer required"):
        list(
            run_tool_calling_llm(
                llm, {"messages": [{"role": "assistant", "content": ""}]}
            )
        )


def test_reset_clears_pending_tool_calls():
    """reset() drops queued tool calls along with the message history.

    Otherwise a reset conversation replays a stale call from the previous
    one on the next chat() — executing code the user never asked for.
    """
    from interpreter.core.core import OpenInterpreter

    interpreter = OpenInterpreter.__new__(OpenInterpreter)
    interpreter.computer = SimpleNamespace(terminate=lambda: None)
    interpreter.messages = [{"role": "user", "content": "hi"}]
    interpreter.last_messages_count = 5
    interpreter._pending_tool_calls = [{"id": "stale"}]

    interpreter.reset()

    assert interpreter.messages == []
    assert interpreter._pending_tool_calls == []


def test_changed_user_messages_clear_stale_queued_calls():
    """Queued calls are served on an unchanged retry, cleared on new input.

    The queue snapshot records the messages it was queued for. Repeating the
    same request replays the queue without an LLM call; appending a new user
    message means the conversation moved on, so the stale calls are dropped
    and the LLM is consulted.
    """
    llm = _make_llm()

    def chat(**params):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="execute",
                                    arguments='{"language": "python", "code": "1"}',
                                ),
                            ),
                            SimpleNamespace(
                                id="call_2",
                                function=SimpleNamespace(
                                    name="execute",
                                    arguments='{"language": "python", "code": "2"}',
                                ),
                            ),
                        ]
                    }
                }
            ]
        }

    llm.completions = chat
    messages = [{"role": "user", "content": "go"}]
    list(run_tool_calling_llm(llm, {"messages": messages}))
    assert len(llm.interpreter._pending_tool_calls) == 1

    def fresh_chat(**params):
        yield {"choices": [{"delta": {"content": "fresh"}}]}

    llm.completions = fresh_chat
    llm.interpreter.messages = [{"role": "user", "content": "never mind"}]
    chunks = list(
        run_tool_calling_llm(
            llm, {"messages": messages + [{"role": "user", "content": "never mind"}]}
        )
    )
    assert llm.interpreter._pending_tool_calls == []
    assert chunks == [{"type": "message", "content": "fresh"}]


def test_computer_image_does_not_clear_queued_calls():
    """A normalized computer image must not look like a new user turn.

    convert_to_openai_messages turns the image message into role user, but the
    raw interpreter message is role computer, so the queued call still belongs
    to the current turn and must be served.
    """
    llm = _make_llm()
    snapshot = [{"role": "user", "content": "go"}]
    llm.interpreter._pending_tool_calls = [
        {
            "id": "q",
            "name": "execute",
            "arguments": '{"language": "python", "code": "print(1)"}',
            "messages": snapshot,
        }
    ]
    llm.interpreter.messages = [
        {"role": "computer", "type": "image", "format": "path", "content": "x.png"}
    ]

    chunks = list(
        run_tool_calling_llm(
            llm, {"messages": [{"role": "user", "content": "image"}]}
        )
    )

    assert any(c.get("type") == "code" for c in chunks)
    assert llm.interpreter._pending_tool_calls == []
