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
        {"type": "code", "format": "python", "content": "print(1)"},
        {"type": "review", "format": "safe", "content": "This is safe"},
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
        {"type": "message", "content": "hello"}
    ]


def test_message_chunk_carries_reasoning_delta():
    """Reasoning deltas ride along on tool-path message chunks for history accumulation (issue #215)."""
    llm = _make_llm([_chunk({"content": "hi", "reasoning_content": "because"})])
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "message", "content": "hi", "reasoning_content": "because"}
    ]


def test_reasoning_only_delta_yields_standalone_message():
    """A reasoning-only tool-path delta is emitted, not gated on content (issue #215)."""
    llm = _make_llm([_chunk({"content": None, "reasoning_content": "thinking"})])
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "message", "content": "", "reasoning_content": "thinking"}
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
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "message", "content": "", "reasoning_content": "r"}
    ]


def test_narration_sharing_a_later_delta_with_a_tool_call_is_a_message():
    """Narration packed with a tool-call continuation is streamed as a message, not a review.

    The verdict-tag classification must run for every delta carrying both
    content and tool calls: untagged narration in a later mixed delta must not
    inherit review status from the earlier tool call that set
    function_call_detected.
    """
    llm = _make_llm(
        [
            _chunk({"tool_calls": [_tool_call("execute", '{"language": "python", "code": "pri')]}),
            _chunk(
                {
                    "content": "And here's why.",
                    "tool_calls": [_tool_call("execute", 'nt(1)"}')],
                }
            ),
        ]
    )
    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "pri"},
        {"type": "message", "content": "And here's why."},
        {"type": "code", "format": "python", "content": "nt(1)"},
    ]


def test_tool_calls_empty_function_fields_skipped():
    """An entry with empty name and arguments does not hide a later valid one."""
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        SimpleNamespace(
                            function=SimpleNamespace(name="", arguments="")
                        ),
                        _tool_call(
                            "execute",
                            '{"language": "python", "code": "print(1)"}',
                        ),
                    ]
                }
            )
        ]
    )

    assert list(run_tool_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"}
    ]


def test_open_review_body_in_later_mixed_delta_stays_a_review():
    """A review opened earlier stays a review when its body shares a mixed delta."""
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call(
                            "execute",
                            '{"language": "python", "code": "print(1)"}',
                        )
                    ]
                }
            ),
            _chunk({"content": "<safe>Looks "}),
            _chunk(
                {
                    "content": "good</safe>",
                    "tool_calls": [_tool_call("execute", '{"language": "python", "code": "print(2)"}')],
                }
            ),
        ]
    )

    result = list(run_tool_calling_llm(llm, {"messages": []}))

    assert any(
        chunk.get("type") == "review" and chunk.get("content") == "Looks "
        for chunk in result
    )
    assert any(
        chunk.get("type") == "review" and chunk.get("content") == "good"
        for chunk in result
    )
    assert not any(
        chunk.get("type") == "message" and "good" in str(chunk.get("content"))
        for chunk in result
    )


def test_non_string_code_in_arguments_does_not_crash():
    """A model that nests `code` as JSON rather than text must not crash the stream.

    Some models emit `{"language": "python", "code": {"nested": 1}}` instead of
    a code string. Slicing that dict against the emitted-so-far string raised
    TypeError and killed the turn, so the arguments are skipped instead of
    crashing.
    """
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call(
                            "execute",
                            '{"language": "python", "code": {"nested": 1}}',
                        )
                    ]
                }
            )
        ]
    )

    assert list(run_tool_calling_llm(llm, {"messages": []})) == []

def test_non_string_code_after_streamed_code_does_not_execute_the_partial_code():
    """A call whose final `code` value is not a string must fail instead of running the partial.

    Arguments fragments are concatenated, and duplicate keys resolve to the last
    value, so a stream that sends `{"code": "print(1)"` and then `, "code": {}}`
    leaves a non-string final value while the earlier string has already been
    yielded. Skipping that final value used to let the superseded partial run as
    if it were the completed code; the turn now raises before execution.
    """
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call(
                            "execute",
                            '{"language": "python", "code": "print(1)"',
                        )
                    ]
                }
            ),
            _chunk(
                {
                    "tool_calls": [
                        _tool_call("execute", ', "code": {"nested": 1}}')
                    ]
                }
            ),
        ]
    )

    chunks = []
    with pytest.raises(TypeError, match="non-string `code`"):
        for chunk in run_tool_calling_llm(llm, {"messages": []}):
            chunks.append(chunk)

    assert any(
        chunk.get("type") == "code" and chunk.get("content") == "print(1)"
        for chunk in chunks
    )

def test_empty_code_after_streamed_code_still_rejects_a_non_string_end():
    """An empty `code` fragment must not disguise that code was already streamed.

    Duplicate keys resolve to the last value, so `{"code": "print(1)"`,
    `, "code": ""`, `, "code": {}}` leaves both a non-string final value and an
    empty current `code`. Testing only the latest value saw "nothing emitted"
    and skipped the error while the superseded `print(1)` chunk remained queued
    for execution; the guard must remember that a non-empty chunk was yielded.
    """
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call(
                            "execute", '{"language": "python", "code": "print(1)"'
                        )
                    ]
                }
            ),
            _chunk({"tool_calls": [_tool_call("execute", ', "code": ""')]}),
            _chunk({"tool_calls": [_tool_call("execute", ', "code": {}}')]}),
        ]
    )

    chunks = []
    with pytest.raises(TypeError, match="non-string `code`"):
        for chunk in run_tool_calling_llm(llm, {"messages": []}):
            chunks.append(chunk)

    assert any(
        chunk.get("type") == "code" and chunk.get("content") == "print(1)"
        for chunk in chunks
    )

def test_recovered_code_that_replaces_streamed_code_is_rejected():
    """A string after a non-string value must extend the emitted code, not rewrite it.

    Duplicate keys resolve to the last value. With fragments
    `{"code": "print(1)"`, `, "code": {}`, `, "code": "print(2)"}` the final
    value is a string again, but an equal-length rewrite yields an empty delta,
    so the earlier `print(1)` would execute while the model's last value was
    `print(2)`. A recovered string that does not start with the code already
    streamed now fails the turn instead.
    """
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call(
                            "execute", '{"language": "python", "code": "print(1)"'
                        )
                    ]
                }
            ),
            _chunk({"tool_calls": [_tool_call("execute", ', "code": {}')]}),
            _chunk({"tool_calls": [_tool_call("execute", ', "code": "print(2)"}')]}),
        ]
    )

    chunks = []
    with pytest.raises(TypeError, match="replaced code already streamed"):
        for chunk in run_tool_calling_llm(llm, {"messages": []}):
            chunks.append(chunk)

    assert any(
        chunk.get("type") == "code" and chunk.get("content") == "print(1)"
        for chunk in chunks
    )
    assert not any(
        chunk.get("type") == "code" and chunk.get("content") == "print(2)"
        for chunk in chunks
    )


def test_recovered_code_that_extends_streamed_code_is_accepted():
    """A string that appends to the emitted code after a non-string value completes normally.

    Deferring the non-string check exists so a provider can repair a nested
    `code` object by sending the real string later. That repair is only valid
    when it extends the text already streamed; the appended remainder is
    yielded and the call succeeds.
    """
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call(
                            "execute", '{"language": "python", "code": "print(1)"'
                        )
                    ]
                }
            ),
            _chunk({"tool_calls": [_tool_call("execute", ', "code": {}')]}),
            _chunk(
                {
                    "tool_calls": [
                        _tool_call(
                            "execute", ', "code": "print(1)\\nprint(2)"}'
                        )
                    ]
                }
            ),
        ]
    )

    chunks = list(run_tool_calling_llm(llm, {"messages": []}))

    assert [
        chunk["content"]
        for chunk in chunks
        if chunk.get("type") == "code"
    ] == ["print(1)", "\nprint(2)"]


def test_empty_code_between_streamed_and_replacement_is_rejected():
    """An empty duplicate `code` must not erase the memory of streamed text.

    The fragments `{"code": "print(1)"`, `, "code": ""`, `, "code": {}`,
    `, "code": "print(2)"}` reset the latest parsed `code` to ``""`` before the
    non-string value and then to `"print(2)"`. Comparing the recovered string
    against the latest value made `startswith("")` pass, so `print(2)` was
    emitted after `print(1)`. Validation must use the text already streamed,
    which rejects the rewrite instead of executing `print(1)print(2)`.
    """
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call(
                            "execute", '{"language": "python", "code": "print(1)"'
                        )
                    ]
                }
            ),
            _chunk({"tool_calls": [_tool_call("execute", ', "code": ""')]}),
            _chunk({"tool_calls": [_tool_call("execute", ', "code": {}')]}),
            _chunk({"tool_calls": [_tool_call("execute", ', "code": "print(2)"}')]}),
        ]
    )

    chunks = []
    with pytest.raises(TypeError, match="replaced code already streamed"):
        for chunk in run_tool_calling_llm(llm, {"messages": []}):
            chunks.append(chunk)

    assert any(
        chunk.get("type") == "code" and chunk.get("content") == "print(1)"
        for chunk in chunks
    )
    assert not any(
        chunk.get("type") == "code" and chunk.get("content") == "print(2)"
        for chunk in chunks
    )


def test_extension_after_empty_and_non_string_values_is_accepted():
    """Recovery still works when empty and non-string duplicates precede the repair.

    A provider can clear `code` with ``""``, send a nested object, then repair
    with the full string. That is a valid continuation when the repaired string
    extends the text already streamed; only the appended remainder is yielded.
    """
    llm = _make_llm(
        [
            _chunk(
                {
                    "tool_calls": [
                        _tool_call(
                            "execute", '{"language": "python", "code": "print(1)"'
                        )
                    ]
                }
            ),
            _chunk({"tool_calls": [_tool_call("execute", ', "code": ""')]}),
            _chunk({"tool_calls": [_tool_call("execute", ', "code": {}')]}),
            _chunk(
                {
                    "tool_calls": [
                        _tool_call(
                            "execute", ', "code": "print(1)\\nprint(2)"}'
                        )
                    ]
                }
            ),
        ]
    )

    chunks = list(run_tool_calling_llm(llm, {"messages": []}))

    assert [
        chunk["content"]
        for chunk in chunks
        if chunk.get("type") == "code"
    ] == ["print(1)", "\nprint(2)"]
