from types import SimpleNamespace

import pytest

from interpreter.core.llm.run_tool_calling_llm import run_tool_calling_llm


class Lang:
    name = "Python"


def _make_llm(chunks, verbose=False, captured=None):
    def completions(**params):
        if captured is not None:
            captured.update(params)
        for chunk in chunks:
            yield chunk

    return SimpleNamespace(
        completions=completions,
        interpreter=SimpleNamespace(
            computer=SimpleNamespace(
                terminal=SimpleNamespace(languages=[Lang()])
            ),
            verbose=verbose,
        ),
    )


def _chunk(delta):
    """Wrap a delta dict in the OpenAI streaming chunk shape the generator consumes."""
    return {"choices": [{"delta": delta}]}


def _tool_call(name, arguments):
    """Build a tool_calls delta entry as the API returns it (an object with .function)."""
    return SimpleNamespace(
        id="toolu_1",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_tool_calling_llm_sets_language_enum():
    """The execute tool's language enum lists every terminal language the computer supports."""
    captured = {}
    llm = _make_llm([], captured=captured)
    list(run_tool_calling_llm(llm, {"messages": []}))
    tool = captured["tools"][0]["function"]
    assert tool["name"] == "execute"
    enum = tool["parameters"]["properties"]["language"]["enum"]
    assert enum == ["python"]


def test_plain_text_content_yields_message_chunks():
    """Streamed text deltas without tool calls are yielded as assistant message chunks."""
    llm = _make_llm([_chunk({"content": "Hello"}), _chunk({"content": " world"})])
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "message", "content": "Hello"},
        {"type": "message", "content": " world"},
    ]


def test_chunks_without_choices_are_skipped():
    """Chunks with no choices list (or an empty one) are ignored rather than failing."""
    llm = _make_llm([{"foo": "bar"}, {"choices": []}, _chunk({"content": "hi"})])
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "message", "content": "hi"}
    ]


def test_legacy_python_tool_call_yields_raw_arguments_as_code():
    """A tool call named 'python' is treated as code in the python language, emitting the raw arguments string."""
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call("python", '{"language":"python","code":"print(1)"}')
                    ]
                }
            )
        ]
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": '{"language":"python","code":"print(1)"}'}
    ]


def test_execute_tool_call_parses_arguments_into_code():
    """An execute tool call's arguments are parsed as JSON and streamed as language-formatted code deltas."""
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call("execute", '{"language": "python", "code": "print(1)"}')
                    ]
                }
            )
        ]
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"}
    ]


def test_streaming_arguments_yield_incremental_code_deltas():
    """Arguments arriving across multiple chunks are merged and only the new characters are yielded."""
    llm = _make_llm(
        [
            _chunk({"tool_calls": [_tool_call("execute", '{"language": "python", "code": "pri')]}),
            _chunk({"tool_calls": [_tool_call("execute", 'nt(1)"}')]}),
        ]
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "pri"},
        {"type": "code", "format": "python", "content": "nt(1)"},
    ]


def test_review_layer_yields_safe_review_after_code():
    """Content following a tool call with a <safe> tag is streamed as a review chunk tagged safe."""
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call("execute", '{"language": "python", "code": "print(1)"}')
                    ]
                }
            ),
            _chunk({"content": "<safe>"}),
            _chunk({"content": "This code is fine"}),
            _chunk({"content": "</safe>"}),
        ]
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"},
        {"type": "review", "format": "safe", "content": ""},
        {"type": "review", "format": "safe", "content": "This code is fine"},
        {"type": "review", "format": "safe", "content": ""},
    ]


def test_review_layer_detects_unsafe_tag():
    """Content following a tool call with an <unsafe> tag is streamed as an unsafe review."""
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call("execute", '{"language": "python", "code": "print(1)"}')
                    ]
                }
            ),
            _chunk({"content": "<unsafe>"}),
            _chunk({"content": "DANGEROUS"}),
        ]
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"},
        {"type": "review", "format": "unsafe", "content": ""},
        {"type": "review", "format": "unsafe", "content": "DANGEROUS"},
    ]


def test_review_layer_detects_warning_tag():
    """Content following a tool call with a <warning> tag is streamed as a warning review."""
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call("execute", '{"language": "python", "code": "print(1)"}')
                    ]
                }
            ),
            _chunk({"content": "<warning>"}),
            _chunk({"content": "careful"}),
        ]
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"},
        {"type": "review", "format": "warning", "content": ""},
        {"type": "review", "format": "warning", "content": "careful"},
    ]


def test_review_content_in_single_chunk_is_buffered_not_yielded():
    """When the full '<safe>text</safe>' arrives in one chunk it enters the review buffer and is not streamed."""
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call("execute", '{"language": "python", "code": "print(1)"}')
                    ]
                }
            ),
            _chunk({"content": "<safe>This is safe</safe>"}),
        ]
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"},
    ]


def test_unparseable_arguments_are_skipped():
    """When tool arguments are not valid JSON, nothing is emitted and the stream continues."""
    llm = _make_llm([_chunk({"tool_calls": [_tool_call("execute", "not json")]})])
    assert list(run_tool_calling_llm(llm, {"messages": []})) == []


def test_unparseable_arguments_verbose_prints_warning(capsys):
    """In verbose mode, unparseable tool arguments trigger the 'Arguments not a dict.' notice."""
    llm = _make_llm(
        [_chunk({"tool_calls": [_tool_call("execute", "not json")]})], verbose=True
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == []
    assert "Arguments not a dict." in capsys.readouterr().out


def test_auth_requires_review_layer(monkeypatch):
    """With INTERPRETER_REQUIRE_AUTHENTICATION, a code turn with no review raises."""
    monkeypatch.setenv("INTERPRETER_REQUIRE_AUTHENTICATION", "true")
    llm = _make_llm([_chunk({"tool_calls": [_tool_call("execute", "{}")]})])
    with pytest.raises(Exception, match="Judge layer required but did not run."):
        list(run_tool_calling_llm(llm, {"messages": []}))


def test_auth_no_tool_call_does_not_raise(monkeypatch):
    """With INTERPRETER_REQUIRE_AUTHENTICATION, a plain-text turn (no tool call) is fine."""
    monkeypatch.setenv("INTERPRETER_REQUIRE_AUTHENTICATION", "true")
    llm = _make_llm([_chunk({"content": "hello"})])
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "message", "content": "hello"}
    ]


# Parallel tool calls (issue #415): helpers that normalize both the object
# shape the OpenAI SDK returns and the dict shape some proxies emit.

from interpreter.core.llm.run_tool_calling_llm import (
    _function_name_and_arguments,
    _queue_extra_tool_call,
    _queued_tool_call_chunk,
    _tool_call_entry_function,
    _tool_call_entry_key,
)


def _dict_entry(index=None, name="browser", arguments='{"url": "x"}', cid="call_2"):
    entry = {"function": {"name": name, "arguments": arguments}, "id": cid}
    if index is not None:
        entry["index"] = index
    return entry


def test_entry_function_reads_object_shape():
    """The SDK object shape exposes .function."""
    assert _tool_call_entry_function(_tool_call("a", "{}")) is not None


def test_entry_function_reads_dict_shape():
    """Proxy implementations return dicts; the function payload is still found."""
    assert _tool_call_entry_function(_dict_entry())["name"] == "browser"


def test_function_name_and_arguments_reads_dict_shape():
    """Name and arguments are read from a dict function payload as well as an object."""
    name, arguments = _function_name_and_arguments(
        {"name": "browser", "arguments": '{"url": "x"}'}
    )
    assert (name, arguments) == ("browser", '{"url": "x"}')


def test_entry_key_prefers_index_over_id():
    """Streamed chunks repeat the index but only carry the id once, so index leads."""
    assert _tool_call_entry_key(_dict_entry(index=1), 0) == ("index", 1)


def test_entry_key_falls_back_to_id():
    """A non-streamed entry with an id but no index keeps that id."""
    assert _tool_call_entry_key(_dict_entry(index=None, cid="call_9"), 0) == "call_9"


def test_entry_key_falls_back_to_position():
    """No index and no id: position is the only stable identity."""
    entry = {"function": {"name": "browser", "arguments": "{}"}}
    assert _tool_call_entry_key(entry, 3) == ("pos", 3)


def test_queue_extra_tool_call_stashes_call():
    """A parallel call the pipeline cannot run this turn is queued on the interpreter."""
    llm = _make_llm([])
    _queue_extra_tool_call(llm, _dict_entry(), ("index", 1))

    queued = llm.interpreter._pending_tool_calls
    assert len(queued) == 1
    assert queued[0]["name"] == "browser"


def test_queue_extra_tool_call_concatenates_fragments():
    """Fragments of one call across chunks share a key, so arguments accumulate."""
    llm = _make_llm([])
    _queue_extra_tool_call(llm, _dict_entry(arguments='{"url":'), ("index", 1))
    _queue_extra_tool_call(llm, _dict_entry(arguments='"x"}'), ("index", 1))

    queued = llm.interpreter._pending_tool_calls
    assert len(queued) == 1
    assert queued[0]["arguments"] == '{"url":"x"}'


def test_queue_extra_tool_call_reuses_existing_queue():
    """The queue is created once and appended to across turns."""
    llm = _make_llm([])
    _queue_extra_tool_call(llm, _dict_entry(index=1), ("index", 1))
    _queue_extra_tool_call(llm, _dict_entry(index=2, cid="call_3"), ("index", 2))
    assert len(llm.interpreter._pending_tool_calls) == 2


def test_queue_extra_tool_call_skips_functionless_entry():
    """An entry with no function payload cannot be run, so it is not queued."""
    llm = _make_llm([])
    _queue_extra_tool_call(llm, {"id": "call_1"}, ("pos", 0))
    assert getattr(llm.interpreter, "_pending_tool_calls", []) == []


def test_queued_tool_call_chunk_replays_as_stream_chunk():
    """A queued call is replayed in the chunk shape the streaming loop consumes."""
    chunk = _queued_tool_call_chunk(
        {"id": "call_1", "name": "browser", "arguments": '{"url": "x"}'}
    )
    entry = chunk["choices"][0]["delta"]["tool_calls"][0]
    assert entry.function.name == "browser"
    assert entry.function.arguments == '{"url": "x"}'
