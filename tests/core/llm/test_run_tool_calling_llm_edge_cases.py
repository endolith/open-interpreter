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

class _DeltaShaped:
    """Mimics litellm's Delta: dict() and [] work, .items() does not.

    Real streaming deltas are Delta objects, verified by probing litellm
    streaming output: dict(delta) and delta.get() work, delta.items() and
    delta.keys() raise AttributeError.
    """

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def __contains__(self, key):
        return key in self._mapping

    def __getitem__(self, key):
        return self._mapping[key]

    def get(self, key, default=None):
        return self._mapping.get(key, default)

    def keys(self):
        return self._mapping.keys()


def test_tool_calls_fallback_handles_delta_objects():
    """A function-less Delta-object chunk is skipped without AttributeError (issue #267).

    The no-usable-function fallback must not iterate delta.items(), which
    real litellm Delta objects lack; dict(delta) works on both shapes.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": _DeltaShaped(
                        {"tool_calls": [SimpleNamespace(function=None)]}
                    )
                }
            ]
        }

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": []})) == []


def test_judge_verdict_single_chunk_is_not_dropped():
    """A <safe>verdict</safe> arriving in one chunk is still yielded (issue #219).

    The review state machine buffered the verdict waiting for a following
    chunk that never comes, then the generator ended and the verdict was
    silently lost.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="execute",
                                    arguments='{"language": "python", "code": "print(1)"}',
                                )
                            )
                        ]
                    }
                }
            ]
        }
        yield {"choices": [{"delta": {"content": "<safe>Looks fine</safe>"}}]}

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"},
        {"type": "review", "format": "safe", "content": "Looks fine"},
    ]


def test_tool_calls_nameless_argumentless_entry_skipped():
    """An entry with neither a name nor arguments does not hide a later valid entry.

    A present-but-empty function object is truthy, so the selection loop used
    to pick it and never reach a valid `execute` entry later in the same list.
    Arguments-only continuations (name=None, arguments set) must still be
    selected, so only the doubly-empty entry is skipped.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            SimpleNamespace(
                                function=SimpleNamespace(name=None, arguments=None)
                            ),
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


def test_double_escaped_arguments_still_produce_a_code_chunk():
    """A provider that escapes JSON twice must not lose the call.

    OpenInference-served DeepSeek streams arguments with one escape level too
    many (#566), so the parsed keys come back quoted (`'"language"'`), every
    lookup missed, and no code chunk was ever yielded — the turn was wasted
    and the task could fail. The repaired object yields the same chunk a
    cleanly-encoded call would.
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
                                    arguments='{"code": "echo hi", "\\"language\\"": "\\"bash\\""}',
                                ),
                            )
                        ]
                    }
                }
            ]
        }

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "bash", "content": "echo hi"}
    ]


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
