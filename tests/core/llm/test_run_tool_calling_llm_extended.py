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
