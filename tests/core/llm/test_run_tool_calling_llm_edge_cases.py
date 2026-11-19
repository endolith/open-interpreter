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

def test_non_string_tool_call_arguments_do_not_end_the_stream():
    """A decoded `function_call.arguments` must not swallow later chunks.

    Some providers stream the arguments of a legacy `function_call` as a
    JSON value rather than a string. Feeding that to `parse_partial_json`
    raises `TypeError` in its character loop, which ended the generator and
    dropped the rest of the turn. The guard skips that chunk so a following
    content delta is still delivered.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "function_call": {
                            "name": "execute",
                            "arguments": [1, 2],
                        }
                    }
                }
            ]
        }
        yield {"choices": [{"delta": {"content": "stream continues"}}]}

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "message", "content": "stream continues"}
    ]


@pytest.mark.parametrize(
    "code_value",
    ['{"nested": 1}', "[1, 2]"],
    ids=["object", "list"],
)
def test_non_string_code_argument_does_not_crash_the_stream(code_value):
    """A `code` value that is not a string must be skipped, not sliced.

    `parse_partial_json` returns whatever shape the model streamed, so a
    model that starts streaming `"code": {` or `"code": [` can hand back an
    object or a list. Indexing that with a slice raised `KeyError: slice(...)`
    and killed the generator mid-turn. Skipping the chunk lets the stream
    keep running; nothing executable can be yielded from a non-string value.
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
                                    arguments='{"language": "python", "code": %s}'
                                    % code_value,
                                ),
                            )
                        ]
                    }
                }
            ]
        }

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": []})) == []


def test_string_code_argument_still_yields_the_code_chunk():
    """The non-string guard must not swallow ordinary string `code` values.

    Paired with the skip test above: if the guard were written the other way
    round (only yielding when the value is *not* a string) this chunk would
    disappear, so assert the normal path still emits the code.
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
                                    arguments='{"language": "python", "code": "print(1)"}',
                                ),
                            )
                        ]
                    }
                }
            ]
        }

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"}
    ]


def test_rejected_list_arguments_do_not_poison_later_string_fragments():
    """A list `arguments` delta must not break the following string fragment.

    The parse guard only skips the chunk; `merge_deltas` had already stored
    the list in `accumulated_deltas`, so the next string fragment evaluated
    `[1, 2] + "..."` and raised `TypeError`, ending the stream before the
    code chunk could be yielded. Rejecting non-string values before the
    accumulator lets the string fragment start the value instead.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "function_call": {
                            "name": "execute",
                            "arguments": [1, 2],
                        }
                    }
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "function_call": {
                            "arguments": '{"language": "python", "code": "print(1)"}',
                        }
                    }
                }
            ]
        }

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"}
    ]


def test_rejected_list_tool_call_arguments_do_not_poison_later_fragments():
    """The same list-arguments guard covers the converted `tool_calls` shape.

    The accumulator sees the call after it is rebuilt from `tool_calls`, so
    a decoded list there took the same `[...] + "..."` path as the legacy
    `function_call` form. A string continuation must still be emitted as
    code.
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
                                    arguments=[1, 2],
                                ),
                            )
                        ]
                    }
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            SimpleNamespace(
                                id="toolu_1",
                                function=SimpleNamespace(
                                    arguments='{"language": "python", "code": "print(1)"}',
                                ),
                            )
                        ]
                    }
                }
            ]
        }

    llm.completions = chat

    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"}
    ]
