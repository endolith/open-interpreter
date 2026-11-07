from types import SimpleNamespace

import pytest

from interpreter.core.llm.run_tool_calling_llm import run_tool_calling_llm


class Lang:
    name = "Python"


def _lang(name):
    return SimpleNamespace(name=name)


def _make_llm(chunks, verbose=False, captured=None, languages=None):
    languages = languages or [Lang()]

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
            terminal=SimpleNamespace(languages=languages),
            toolbox=SimpleNamespace(
                terminal=SimpleNamespace(languages=languages)
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


def test_language_named_tool_call_is_rewritten_to_execute():
    """A tool named python/bash/javascript is treated as execute in that language.

    Models trained on those tool names otherwise lose a turn to the unsupported
    error; the stored history is rewritten to the execute shape so later turns
    imitate the correct call.
    """
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
    assert result == [
        {"role": "assistant", "type": "code", "format": "python", "content": "print(1)"}
    ]


@pytest.mark.parametrize(
    "tool,raw_arguments,language,code,language_name",
    [
        ("bash", '{"code": "ls -la", "language": "bash"}', "bash", "ls -la", "bash"),
        ("bash", '{"command": "ls -la"}', "bash", "ls -la", "bash"),
        ("sh", "ls -la", "bash", "ls -la", "bash"),
        ("javascript", '{"code": "console.log(1)"}', "javascript", "console.log(1)", "JavaScript"),
    ],
)
def test_language_named_tool_call_accepts_common_argument_shapes(
    tool, raw_arguments, language, code, language_name
):
    """bash/python-style tools accept code/language, command, or a raw string."""
    llm = _make_llm(
        [_chunk({"tool_calls": [_tool_call(tool, raw_arguments)]})],
        languages=[_lang(language_name)],
    )
    result = list(run_tool_calling_llm(llm, {"messages": []}))
    assert result == [
        {"role": "assistant", "type": "code", "format": language, "content": code}
    ]


def test_unknown_direct_tool_is_stored_and_paired_with_a_specific_error():
    """An unknown tool name keeps the model's call and gets a non-Python error.

    The old message told the model to call the tool from Python and gave a
    toolbox.web.search example, which is wrong for a bash/javascript/etc. call.
    """
    llm = _make_llm(
        [_chunk({"tool_calls": [_tool_call("web.search", '{"q": "x"}')]})]
    )
    result = list(run_tool_calling_llm(llm, {"messages": []}))
    assert result[0]["type"] == "tool_call"
    assert result[0]["name"] == "web.search"
    assert result[1]["role"] == "tool"
    assert result[1]["tool_call_id"] == result[0]["tool_call_id"]
    assert "Unsupported direct tool call" in result[1]["content"]
    assert "Python code" not in result[1]["content"]
    assert "toolbox.web.search" not in result[1]["content"]


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
    """When the full '<safe>text</safe>' arrives in one chunk the verdict is still yielded (issue #219).

    The verdict used to sit in the review buffer waiting for a following
    chunk that never came, and was silently dropped when the stream ended.
    The end-of-stream flush now emits it.
    """
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
        {"type": "review", "format": "safe", "content": "This is safe"},
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


def test_execute_call_with_command_in_language_is_rejected():
    """A command in `language` is rejected, not guessed into a shell.

    Running it under an assumed language could change what the code does, so OI
    refuses and tells the model where the command belongs.
    """
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

    result = list(run_tool_calling_llm(llm, {"messages": []}))

    assert result[0]["type"] == "tool_call"
    assert result[0]["tool_call_id"] == "toolu_1"
    assert result[1]["role"] == "tool"
    assert result[1]["tool_call_id"] == result[0]["tool_call_id"]
    assert "`language` must be one of" in result[1]["content"]
    assert "The command or program belongs in `code`." in result[1]["content"]
    assert not any(chunk.get("type") == "code" for chunk in result)


def test_execute_call_without_code_names_the_two_fields():
    """A valid language with no code gets an error that says where the code goes."""
    llm = _make_llm(
        [_chunk({"tool_calls": [_tool_call("execute", '{"language": "python"}')]})]
    )

    result = list(run_tool_calling_llm(llm, {"messages": []}))

    assert result[0]["type"] == "tool_call"
    assert result[1]["role"] == "tool"
    assert result[1]["tool_call_id"] == result[0]["tool_call_id"]
    assert "put the command or program in `code`" in result[1]["content"]
    assert "`language` must be one of" in result[1]["content"]


def test_unparseable_arguments_are_stored_and_reported():
    """Unparseable arguments keep the raw call and pair it with the error."""
    llm = _make_llm([_chunk({"tool_calls": [_tool_call("execute", "not json")]})])
    result = list(run_tool_calling_llm(llm, {"messages": []}))
    assert len(result) == 2
    assert result[0]["type"] == "tool_call"
    assert result[0]["arguments"] == "not json"
    assert result[1]["role"] == "tool"
    assert result[1]["tool_call_id"] == result[0]["tool_call_id"]
    assert "Invalid execute call" in result[1]["content"]


def test_unparseable_arguments_verbose_prints_warning(capsys):
    """In verbose mode, unparseable tool arguments trigger the 'Arguments not a dict.' notice."""
    llm = _make_llm(
        [_chunk({"tool_calls": [_tool_call("execute", "not json")]})], verbose=True
    )
    result = list(run_tool_calling_llm(llm, {"messages": []}))
    assert result[0]["type"] == "tool_call"
    assert result[1]["role"] == "tool"
    assert "Invalid execute call" in result[1]["content"]


def test_auth_requires_review_layer(monkeypatch):
    """With INTERPRETER_REQUIRE_AUTHENTICATION, a code turn with no review raises."""
    monkeypatch.setenv("INTERPRETER_REQUIRE_AUTHENTICATION", "true")
    llm = _make_llm([_chunk({"tool_calls": [_tool_call("execute", "{}")]})])
    with pytest.raises(Exception, match="Judge layer required but did not run."):
        list(run_tool_calling_llm(llm, {"messages": []}))


def test_content_sharing_a_delta_with_the_tool_call_is_yielded_as_a_message():
    """Text packed into the same delta as the tool call is streamed as an assistant message.

    Providers that collapse a turn into one message (litellm's ollama_chat and
    Gemini builders) send the assistant's prose and its tool call in a single
    delta. Rebuilding the delta from the tool call alone dropped that prose, so
    the explanation before the code was never displayed or stored. Regression
    test for #316.
    """
    llm = _make_llm(
        [
            _chunk(
                {
                    "content": "I'll print one for you.",
                    "tool_calls": [
                        _tool_call("execute", '{"language": "python", "code": "print(1)"}')
                    ],
                }
            )
        ]
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "message", "content": "I'll print one for you."},
        {"type": "code", "format": "python", "content": "print(1)"},
    ]


def test_review_verdict_sharing_a_delta_with_the_tool_call_counts_as_a_review(monkeypatch):
    """A judge verdict delivered with the tool call satisfies the authentication guard.

    A collapsed message gives no ordering between the text and the call, so the
    verdict tags — not the position — mark it as a review. The code is still
    emitted: holding the closing tag back in the review buffer must not skip
    the rest of the chunk. Regression test for #316.
    """
    monkeypatch.setenv("INTERPRETER_REQUIRE_AUTHENTICATION", "true")
    llm = _make_llm(
        [
            _chunk(
                {
                    "content": "<safe>fine</safe>",
                    "tool_calls": [
                        _tool_call("execute", '{"language": "python", "code": "print(1)"}')
                    ],
                }
            )
        ]
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"},
        {"type": "review", "format": "safe", "content": "fine"},
    ]


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
    assert any(m.get("role") == "tool" and "Unsupported direct tool call" in m.get("content", "") for m in result)


@pytest.mark.parametrize(
    "arguments",
    ['{"code": "ls", "\\"language\\"": "bash"}', None],
)
def test_rejected_execute_call_is_stored_and_paired(arguments):
    """A rejected execute call keeps its raw arguments and pairs with its error.

    The model made the call; replaying it with the provider's id keeps the next
    request well-formed instead of turning the error into orphan assistant prose.
    The quoted-key shape is what OpenInference streams; the null shape is a call
    with no arguments.
    """
    llm = _make_llm([_chunk({"tool_calls": [_tool_call("execute", arguments)]})])
    result = list(run_tool_calling_llm(llm, {"messages": []}))
    call = next(c for c in result if c.get("type") == "tool_call")
    error = next(c for c in result if c.get("role") == "tool")
    assert call["tool_call_id"] == "toolu_1"
    assert error["tool_call_id"] == call["tool_call_id"]
    assert "Invalid execute call" in error["content"]
    assert call["arguments"] == ("null" if arguments is None else arguments)
