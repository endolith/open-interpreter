"""Tests for run_tool_calling_llm edge cases not covered by existing tests."""

from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.llm.run_tool_calling_llm import process_messages, run_tool_calling_llm
from interpreter.core.llm.utils.tool_call_errors import TOOL_CALL_ERROR_PREFIX


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


def _one_call_stream(name, arguments):
    """A one-chunk completion stream carrying a single tool call."""

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            SimpleNamespace(
                                id="toolu_1",
                                function=SimpleNamespace(name=name, arguments=arguments),
                            )
                        ]
                    }
                }
            ]
        }

    return chat


def test_unsupported_tool_name_is_reported_instead_of_dropped():
    """A call this build cannot run must leave a trace in the transcript.

    Only `execute` (and its legacy aliases) is understood, but nothing rejected
    an unknown name: arguments without `code` yielded no chunk, nothing was
    appended, and the turn ended as if the model had been silent (#629). The
    error is stored as assistant text so the next turn can correct itself.
    """
    llm = _make_llm()
    llm.completions = _one_call_stream("bash", '{"command": "ls"}')

    chunks = list(run_tool_calling_llm(llm, {"messages": []}))

    assert len(chunks) == 1
    assert chunks[0]["type"] == "message"
    assert chunks[0]["content"].startswith(TOOL_CALL_ERROR_PREFIX)
    assert "`bash`" in chunks[0]["content"]
    assert "execute" in chunks[0]["content"]


def test_execute_without_code_is_reported_instead_of_dropped():
    """`execute` with the wrong argument names is reported, not silently lost.

    A model that calls `execute` with something other than `language` plus
    `code` (here a shell-style `command`) used to get no code chunk and no
    message, ending the turn blank. The message names the missing arguments
    instead.
    """
    llm = _make_llm()
    llm.completions = _one_call_stream("execute", '{"command": "ls"}')

    chunks = list(run_tool_calling_llm(llm, {"messages": []}))

    assert len(chunks) == 1, "a runnable-free call must still produce output"
    assert chunks[0]["type"] == "message"
    assert chunks[0]["content"].startswith(TOOL_CALL_ERROR_PREFIX)
    assert "`execute`" in chunks[0]["content"]
    assert "`language`" in chunks[0]["content"]
    assert "`code`" in chunks[0]["content"]


def test_legacy_function_call_delta_is_reported():
    """A legacy `function_call` delta that cannot run is reported like a tool call.

    Some providers stream the old `function_call` shape, which merges into
    `accumulated_deltas` without setting the tool_calls conversion flag. It
    used to bypass the report and end the turn silently.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "function_call": {"name": "execute", "arguments": "not json"}
                    }
                }
            ]
        }

    llm.completions = chat

    chunks = list(run_tool_calling_llm(llm, {"messages": []}))

    assert len(chunks) == 1
    assert chunks[0]["content"].startswith(TOOL_CALL_ERROR_PREFIX)


def test_buffered_review_is_flushed_before_the_error():
    """A buffered verdict must not displace the error as the transcript tail.

    A verdict that arrives packed with a tool call and ends in its closing tag
    stays buffered until the stream ends. The report is emitted after that
    flush, so `respond` sees the tool-call error as the last message and
    retries. Emitting the report first left the review as the tail and the
    retry never fired.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "content": "<safe>Looks fine</safe>",
                        "tool_calls": [
                            SimpleNamespace(
                                id="toolu_1",
                                function=SimpleNamespace(name="other", arguments='{"x": 1}'),
                            )
                        ],
                    }
                }
            ]
        }

    llm.completions = chat

    chunks = list(run_tool_calling_llm(llm, {"messages": []}))

    assert [chunk["type"] for chunk in chunks] == ["review", "message"]
    assert chunks[-1]["content"].startswith(TOOL_CALL_ERROR_PREFIX)


def test_nameless_argumentless_tool_call_is_reported():
    """A tool-call entry with no name or arguments must not end the turn silently.

    The conversion loop skips a doubly-empty entry because there is nothing
    usable to select, but that left nothing in the transcript and the turn
    ended blank, the same silent failure as #629. The report names the missing
    function so the next turn can correct itself.
    """
    llm = _make_llm()

    def chat(*args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            SimpleNamespace(function=SimpleNamespace(name="", arguments=""))
                        ]
                    }
                }
            ]
        }

    llm.completions = chat

    chunks = list(run_tool_calling_llm(llm, {"messages": []}))

    assert len(chunks) == 1, "an unusable tool call must still produce output"
    assert chunks[0]["type"] == "message"
    assert chunks[0]["content"].startswith(TOOL_CALL_ERROR_PREFIX)
    assert "without a function name" in chunks[0]["content"]
