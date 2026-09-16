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
    """Markdown fenced code blocks are parsed into typed code chunks with language format.

    The fence and the language label are structural markup, not code, so they
    must not appear in the emitted content. This previously yielded a leading
    '```\\n' code chunk whenever the fence and label arrived in one delta,
    which concatenates into the code message that respond.py executes.
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\n"}}]},
            {"choices": [{"delta": {"content": "print(1)\n"}}]},
            {"choices": [{"delta": {"content": "```"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "system"}]}))
    assert result == [
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
    """The stream stops when the closing ``` of a fenced code block is seen.

    Nothing after the closing fence is emitted, and the fence itself is not
    part of the code body.
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\n"}}]},
            {"choices": [{"delta": {"content": "print(1)\n"}}]},
            {"choices": [{"delta": {"content": "```\n"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert result == [
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


def _text_of(chunks):
    """Concatenate the assistant message text from a run_text_llm result."""
    return "".join(c["content"] for c in chunks if c["type"] == "message")


def _code_of(chunks):
    """Concatenate the code body from a run_text_llm result."""
    return "".join(c["content"] for c in chunks if c["type"] == "code")


@pytest.mark.parametrize(
    "chunks,expected",
    [
        (["Run", " `", "ls", "`", " to list files"], "Run `ls` to list files"),
        (["Run ", "`ls`", " now"], "Run `ls` now"),
        (["Use ", "`pip`"], "Use `pip`"),
    ],
)
def test_inline_backticks_in_prose_are_preserved(chunks, expected):
    """Prose containing inline code survives the fence-disambiguation hold-back.

    A trailing backtick may be the start of a ``` fence, so it is held back
    until the next delta settles it. Previously the held-back delta was
    dropped outright, so every backtick-terminated delta vanished from both
    the display and the stored assistant message. Real tokenizers split
    inline code exactly this way ('Run', ' `', 'ls', '`', ...).
    """
    llm = _make_llm([{"choices": [{"delta": {"content": c}}]} for c in chunks])
    assert _text_of(list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))) == expected


def test_whole_code_block_in_one_delta_is_emitted():
    """A fenced block arriving as a single delta still yields its code.

    Non-streaming backends and custom llm.completions implementations deliver
    the whole reply at once. Because such a delta ends with a backtick it was
    held back for fence disambiguation and never flushed, so the generator
    finished having yielded nothing at all: no code, and no execution.
    """
    llm = _make_llm([{"choices": [{"delta": {"content": "```python\nprint(1)\n```"}}]}])
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert result == [{"type": "code", "format": "python", "content": "print(1)\n"}]


@pytest.mark.parametrize(
    "language,body",
    [
        ("python", "print('python rocks')\n"),
        ("r", "print('rrr')\n"),
        ("shell", "echo shell\n"),
    ],
)
def test_code_containing_the_language_name_is_not_mangled(language, body):
    """Code keeps every character even when it contains its own language name.

    The language label was previously removed with content.replace(language, "")
    on each code delta, which deletes the label text wherever it occurs in the
    body — 'r' code lost every letter r. The label is structural, so it is now
    consumed positionally by advancing past the line that holds it.
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": f"```{language}\n"}}]},
            {"choices": [{"delta": {"content": body}}]},
            {"choices": [{"delta": {"content": "```"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert _code_of(result) == body
    assert {c["format"] for c in result} == {language}


def test_prose_before_a_fence_is_emitted_then_code():
    """Text preceding a code fence is emitted as a message before the code starts."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "Sure. "}}]},
            {"choices": [{"delta": {"content": "```python\n"}}]},
            {"choices": [{"delta": {"content": "print(1)\n"}}]},
            {"choices": [{"delta": {"content": "```"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert _text_of(result) == "Sure. "
    assert _code_of(result) == "print(1)\n"


def test_fence_split_across_deltas_yields_the_same_code():
    """The opening fence split from its language label parses identically.

    Providers split '```python' across deltas; the result must not depend on
    where the boundary falls.
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```"}}]},
            {"choices": [{"delta": {"content": "python\nprint(1)\n"}}]},
            {"choices": [{"delta": {"content": "```"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert result == [{"type": "code", "format": "python", "content": "print(1)\n"}]


@pytest.mark.parametrize(
    "chunks,expected",
    [
        (["```py"], "```py"),
        (["```"], "```"),
        (["```", "pyth"], "```pyth"),
    ],
)
def test_a_fence_cut_off_before_its_language_line_is_not_dropped(chunks, expected):
    """A stream that opens a fence but ends before the language line completes.

    The fence and any partial label were consumed into the code-block state but
    the language line never finished, so nothing was emitted and the text was
    lost. This function exists to never silently drop what the model sent, so
    an unconfirmed fence is flushed as literal message content at end of stream.
    """
    llm = _make_llm([{"choices": [{"delta": {"content": c}}]} for c in chunks])
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert "".join(c["content"] for c in result) == expected
