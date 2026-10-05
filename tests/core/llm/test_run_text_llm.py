from types import SimpleNamespace

import pytest

from interpreter.core.llm.run_text_llm import run_text_llm


def _make_llm(chunks, execution_instructions=None, verbose=False, os_mode=False):
    def completions(**params):
        for chunk in chunks:
            yield chunk

    return SimpleNamespace(
        completions=completions,
        execution_instructions=execution_instructions,
        interpreter=SimpleNamespace(verbose=verbose, os=os_mode),
    )


def test_plain_text_yields_messages():
    """Streaming text deltas from the LLM are yielded as assistant message chunks."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "system"}]}))
    assert result == [
        {"type": "message", "content": "Hello"},
        {"type": "message", "content": " world"},
    ]


def test_code_block_yields_code_chunks():
    """Markdown fenced code blocks in the stream are parsed into typed code chunks with language format."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\n"}}]},
            {"choices": [{"delta": {"content": "print(1)\n"}}]},
            {"choices": [{"delta": {"content": "```"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "system"}]}))
    assert result == [
        {"type": "code", "format": "python", "content": "```\n"},
        {"type": "code", "format": "python", "content": "print(1)\n"},
    ]


def test_execution_instructions_appended():
    """When set, execution_instructions are appended to the system message before the API call."""
    llm = _make_llm([], execution_instructions="Run safely.")
    params = {"messages": [{"content": "base"}]}
    list(run_text_llm(llm, params))
    assert params["messages"][0]["content"] == "base\nRun safely."


def test_chunks_without_choices_are_skipped():
    """Chunks with no choices list (or an empty one) are ignored rather than failing."""
    llm = _make_llm([{"foo": "bar"}, {"choices": []}, {"choices": [{"delta": {"content": "hi"}}]}])
    assert list(run_text_llm(llm, {"messages": [{"content": "sys"}]})) == [
        {"type": "message", "content": "hi"}
    ]


def test_verbose_prints_each_chunk(capsys):
    """In verbose mode each raw chunk is printed as it streams."""
    llm = _make_llm(
        [{"choices": [{"delta": {"content": "hi"}}]}], verbose=True
    )
    assert list(run_text_llm(llm, {"messages": [{"content": "sys"}]})) == [
        {"type": "message", "content": "hi"}
    ]
    assert "Chunk in coding_llm" in capsys.readouterr().out


def test_code_block_exit_returns():
    """The stream stops when the closing ``` of a fenced code block is seen."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\n"}}]},
            {"choices": [{"delta": {"content": "print(1)\n"}}]},
            {"choices": [{"delta": {"content": "```\n"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert result == [
        {"type": "code", "format": "python", "content": "```\n"},
        {"type": "code", "format": "python", "content": "print(1)\n"},
    ]


def test_empty_language_defaults_to_python():
    """A code fence with no language label (e.g. '```\\n') defaults to python in text mode."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```\n"}}]},
            {"choices": [{"delta": {"content": "print(1)\n"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert result == [
        {"type": "code", "format": "python", "content": "```\n"},
        {"type": "code", "format": "python", "content": "print(1)\n"},
    ]


def test_none_content_chunk_is_skipped():
    """Chunks whose delta content is None are skipped rather than treated as text."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": None}}]},
            {"choices": [{"delta": {"content": "hi"}}]},
        ]
    )
    assert list(run_text_llm(llm, {"messages": [{"content": "sys"}]})) == [
        {"type": "message", "content": "hi"}
    ]


def test_execution_instructions_unappendable_message_reraises(capsys):
    """A non-string first message with execution_instructions prints context and re-raises the error."""
    llm = _make_llm([], execution_instructions="careful")
    with pytest.raises(TypeError):
        list(run_text_llm(llm, {"messages": [{"content": 123}]}))
    assert "params[\"messages\"][0]" in capsys.readouterr().out


def test_empty_language_yields_nothing_in_os_mode():
    """In OS mode an unlabelled fence currently produces no output at all.

    Worse than a mislabel: neither branch of the default fires when os is true,
    so `language` stays "" and `if language:` suppresses every chunk. The reply
    vanishes silently. See #392; the xfail below describes the intent.
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```\n"}}]},
            {"choices": [{"delta": {"content": "some notes\n"}}]},
        ],
        os_mode=True,
    )

    assert list(run_text_llm(llm, {"messages": [{"content": "sys"}]})) == []


@pytest.mark.xfail(
    reason="#392: the 'text' branch duplicates the condition above it and cannot be taken",
)
def test_empty_language_defaults_to_text_in_os_mode():
    """In OS mode an unlabelled fence should be labelled text, not python.

    The branch that would do this is unreachable: `elif llm.interpreter.os ==
    False` repeats the identical condition from the branch above, so `"text"` is
    never produced. The comment on that branch describes OS mode, so the second
    comparison was likely meant to be `== True`. See #392.
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```\n"}}]},
            {"choices": [{"delta": {"content": "some notes\n"}}]},
        ],
        os_mode=True,
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))

    assert {chunk.get("format") for chunk in result} == {"text"}
    assert result, "the block must not vanish entirely (see #392)"


def test_a_delta_with_no_content_key_yields_an_empty_message():
    """A delta missing the content key is not None, so it is not skipped.

    There is a `if content == None: continue` guard, which catches an explicit
    null but not an absent key: `.get("content", "")` yields "" and the delta is
    emitted as an empty message. The distinction is real — the existing
    none-content test covers the null case, and the default this pins is what
    stops a role-only delta from becoming the string "None".
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {}}]},
            {"choices": [{"delta": {"content": "hi"}}]},
        ]
    )
    assert list(run_text_llm(llm, {"messages": [{"content": "sys"}]})) == [
        {"type": "message", "content": ""},
        {"type": "message", "content": ""},
        {"type": "message", "content": "hi"},
    ]


def test_os_flag_true_does_not_change_labelled_languages():
    """A labelled fence keeps its language in OS mode.

    The OS-mode branch is only reached when the label is empty, so a reply that
    says ```bash must stay bash. This guards the fix for #392 against
    over-reaching into labelled fences.
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```bash\n"}}]},
            {"choices": [{"delta": {"content": "echo hi\n"}}]},
        ],
        os_mode=True,
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))

    assert {chunk.get("format") for chunk in result} == {"bash"}
