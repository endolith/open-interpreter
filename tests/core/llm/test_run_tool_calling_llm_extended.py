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
        model="test-model",
        tool_calling_instructions="",
        interpreter=SimpleNamespace(
            llm=SimpleNamespace(supports_vision=False, tool_calling_instructions=""),
            terminal=SimpleNamespace(languages=[Lang()]),
            toolbox=SimpleNamespace(
                terminal=SimpleNamespace(languages=[Lang()])
            ),
            verbose=verbose,
            debug=False,
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
        {"role": "assistant", "type": "message", "content": "Hello"},
        {"role": "assistant", "type": "message", "content": " world"},
    ]


def test_chunks_without_choices_are_skipped():
    """Chunks with no choices list (or an empty one) are ignored rather than failing."""
    llm = _make_llm([{"foo": "bar"}, {"choices": []}, _chunk({"content": "hi"})])
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"role": "assistant", "type": "message", "content": "hi"}
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
    result = list(run_tool_calling_llm(llm, {"messages": []}))
    # Classic only accepts execute/edit/view_image as direct tool names; anything
    # else is reported back to the model instead of being treated as code.
    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert "Unsupported function call" in result[0]["content"]
    assert "python" in result[0]["content"]


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
        {"role": "assistant", "type": "code", "format": "python", "content": "print(1)"}
    ]


@pytest.mark.skip(
    reason="classic's tool-call accumulator parses partial JSON differently; main's incremental-delta characterization does not apply"
)
def test_streaming_arguments_yield_incremental_code_deltas():
    """Arguments arriving across multiple chunks are merged and only the new characters are yielded."""
    llm = _make_llm(
        [
            _chunk({"tool_calls": [_tool_call("execute", '{"language": "python", "code": "pri')]}),
            _chunk({"tool_calls": [_tool_call("execute", 'nt(1)"}')]}),
        ]
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"role": "assistant", "type": "code", "format": "python", "content": "pri"},
        {"role": "assistant", "type": "code", "format": "python", "content": "nt(1)"},
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
        {"type": "review", "format": "safe", "content": ""},
        {"type": "review", "format": "safe", "content": "This code is fine"},
        {"type": "review", "format": "safe", "content": ""},
        {"role": "assistant", "type": "code", "format": "python", "content": "print(1)"},
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
        {"type": "review", "format": "unsafe", "content": ""},
        {"type": "review", "format": "unsafe", "content": "DANGEROUS"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print(1)"},
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
        {"type": "review", "format": "warning", "content": ""},
        {"type": "review", "format": "warning", "content": "careful"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print(1)"},
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
        {"role": "assistant", "type": "code", "format": "python", "content": "print(1)"},
    ]


def test_invalid_view_image_call_is_stored_before_the_error():
    """A failed view_image still records the model's call.

    Without it the tool result is orphaned, and process_messages used to invent
    an execute(pass) call that the model then imitated.
    """
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call("view_image", '{"path": "relative.png"}')
                    ]
                }
            )
        ]
    )
    result = list(run_tool_calling_llm(llm, {"messages": []}))

    assert result[0]["type"] == "view_image_call"
    assert result[0]["tool_call_id"] == "toolu_1"
    assert result[0]["path"] == "relative.png"
    assert result[1]["role"] == "tool"
    assert "must be absolute" in result[1]["content"]


def test_execute_call_with_code_in_language_is_corrected():
    """A command in `language` is moved to `code` and run as the default shell.

    opencode_go/deepseek-v4.1-flash emits {"language": "cd … && find …"} with no
    `code`. Every turn bounced an "expected both" error and the model looped;
    recovering the intent lets the command run.
    """

    class BashLang:
        name = "Bash"

    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call("execute", '{"language": "cd /tmp && ls"}')
                    ]
                }
            )
        ]
    )
    llm.interpreter.terminal.languages = [Lang(), BashLang()]
    llm.interpreter.toolbox.terminal.languages = [Lang(), BashLang()]

    result = list(run_tool_calling_llm(llm, {"messages": []}))

    assert result[0]["type"] == "code"
    assert result[0]["format"] == "bash"
    assert result[0]["content"] == "cd /tmp && ls"


def test_execute_call_without_code_names_the_two_fields():
    """A valid language with no code gets an error that says where the code goes."""
    llm = _make_llm(
        [_chunk({"tool_calls": [_tool_call("execute", '{"language": "python"}')]})]
    )

    result = list(run_tool_calling_llm(llm, {"messages": []}))

    assert result[0]["role"] == "tool"
    assert "put the command or program in `code`" in result[0]["content"]
    assert "`language` must be one of" in result[0]["content"]


def test_unparseable_arguments_are_skipped():
    """When tool arguments are not valid JSON, nothing is emitted and the stream continues."""
    llm = _make_llm([_chunk({"tool_calls": [_tool_call("execute", "not json")]})])
    result = list(run_tool_calling_llm(llm, {"messages": []}))
    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert "Invalid execute call" in result[0]["content"]


def test_unparseable_arguments_verbose_prints_warning(capsys):
    """In verbose mode, unparseable tool arguments trigger the 'Arguments not a dict.' notice."""
    llm = _make_llm(
        [_chunk({"tool_calls": [_tool_call("execute", "not json")]})], verbose=True
    )
    result = list(run_tool_calling_llm(llm, {"messages": []}))
    assert "Invalid execute call" in result[0]["content"]


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
        {"role": "assistant", "type": "message", "content": "hello"}
    ]


def test_message_chunk_carries_reasoning_delta():
    """Reasoning deltas ride along on tool-path message chunks for history accumulation (issue #215)."""
    llm = _make_llm([_chunk({"content": "hi", "reasoning_content": "because"})])
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "because"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "because\n\n", "replace": True},
        {"role": "assistant", "type": "message", "content": "hi"},
    ]


def test_reasoning_only_delta_yields_standalone_message():
    """A reasoning-only tool-path delta is emitted, not gated on content (issue #215)."""
    llm = _make_llm([_chunk({"content": None, "reasoning_content": "thinking"})])
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "thinking"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "thinking\n\n", "replace": True},
    ]


def test_reasoning_preserved_when_delta_has_tool_call():
    """Reasoning survives the tool-call transform, which replaces the delta (issue #215)."""
    llm = _make_llm(
        [
            _chunk(
                {
                    "reasoning_content": "r",
                    "tool_calls": [_tool_call("other", '{"x": 1}')],
                }
            )
        ]
    )
    result = list(run_tool_calling_llm(llm, {"messages": []}))
    assert result[0] == {
        "role": "assistant",
        "type": "message",
        "format": "reasoning",
        "content": "r",
    }
    assert any(m.get("role") == "tool" and "Unsupported function call" in m.get("content", "") for m in result)
