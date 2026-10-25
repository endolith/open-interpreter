from types import SimpleNamespace

import pytest

from interpreter.core.llm.run_text_llm import _withheld_fence_chars, run_text_llm


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


def test_a_delta_with_no_content_key_is_skipped():
    """A delta missing the content key is skipped like an explicit null.

    The buffer/cursor rewrite treats a missing key the same as None: with
    nothing to append to the buffer there is nothing to emit, so role-only
    deltas no longer produce empty message chunks (fixed in #299).
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {}}]},
            {"choices": [{"delta": {"content": "hi"}}]},
        ]
    )
    assert list(run_text_llm(llm, {"messages": [{"content": "sys"}]})) == [
        {"type": "message", "content": "hi"},
    ]


def test_reasoning_only_delta_yields_message_with_reasoning():
    """A delta with None content but reasoning_content is kept, not skipped (issue #215)."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": None, "reasoning_content": "Let me think"}}]},
            {"choices": [{"delta": {"content": "hi"}}]},
        ]
    )
    assert list(run_text_llm(llm, {"messages": [{"content": "sys"}]})) == [
        {"type": "message", "content": "", "reasoning_content": "Let me think"},
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


def test_reasoning_content_attached_per_chunk():
    """Each message chunk carries its own reasoning delta for history accumulation (issue #215)."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "a", "reasoning_content": "r1"}}]},
            {"choices": [{"delta": {"content": "b", "reasoning_content": "r2"}}]},
        ]
    )
    assert list(run_text_llm(llm, {"messages": [{"content": "sys"}]})) == [
        {"type": "message", "content": "a", "reasoning_content": "r1"},
        {"type": "message", "content": "b", "reasoning_content": "r2"},
    ]


def test_reasoning_survives_split_fence_guard():
    """Reasoning arriving while the fence parser holds a chunk is still emitted (issue #215).

    When accumulated output ends in a backtick the parser `continue`s past the
    chunk, which used to drop any reasoning delta riding along with it.
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "code `"}}]},
            {
                "choices": [
                    {"delta": {"content": None, "reasoning_content": "thinking"}}
                ]
            },
            {"choices": [{"delta": {"content": "more"}}]},
        ]
    )
    assert list(run_text_llm(llm, {"messages": [{"content": "sys"}]})) == [
        {"type": "message", "content": "code "},
        {"type": "message", "content": "", "reasoning_content": "thinking"},
        {"type": "message", "content": "`more"},
    ]


def test_reasoning_inside_code_block_is_emitted():
    """Reasoning arriving mid-code-block is emitted on its own message (issue #215).

    Code chunks carry no reasoning, and the message branch is skipped inside a
    fence, so the reasoning used to never reach stored history.
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\n"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "content": "print(1)\n",
                            "reasoning_content": "r-code",
                        }
                    }
                ]
            },
        ]
    )
    assert list(run_text_llm(llm, {"messages": [{"content": "sys"}]})) == [
        {"type": "message", "content": "", "reasoning_content": "r-code"},
        {"type": "code", "format": "python", "content": "print(1)\n"},
    ]


def test_reasoning_on_closing_fence_is_emitted_before_return():
    """Reasoning riding on the closing fence is emitted before the turn ends.

    The closing fence returns immediately; without this branch the reasoning
    that arrived with it would never reach stored history.
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\n"}}]},
            {"choices": [{"delta": {"content": "print(1)\n"}}]},
            {
                "choices": [
                    {"delta": {"content": "```\n", "reasoning_content": "done"}}
                ]
            },
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert result[-1] == {
        "type": "message",
        "content": "",
        "reasoning_content": "done",
    }


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


def test_prose_and_fence_in_a_single_delta_emit_both():
    """Prose sharing a chunk with its fence is emitted before entering the block."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "Sure. ```python\nprint(1)\n```"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert _text_of(result) == "Sure. "
    assert _code_of(result) == "print(1)\n"


def test_stream_cut_off_inside_a_code_block_flushes_pending_code():
    """A stream ending mid-block (no closing fence) still yields buffered code."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\n"}}]},
            {"choices": [{"delta": {"content": "print(1)\n``"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert _code_of(result) == "print(1)\n``"


@pytest.mark.parametrize(
    "text, expected",
    [
        # No trailing backticks: nothing ambiguous.
        ("", 0),
        ("hello", 0),
        ("a`b", 0),
        # One or two trailing backticks: ambiguous, withheld whole.
        ("`", 1),
        ("ls`", 1),
        ("a`b`", 1),
        ("``", 2),
        ("ls``", 2),
        # Three or more is already a fence, so nothing is withheld — even
        # though a four-backtick run also "ends with three", and "``" is
        # itself a suffix of the fence.
        ("```", 0),
        ("ls```", 0),
        ("````", 0),
        ("`````", 0),
        # A complete fence wins even when backticks follow ordinary text.
        ("```py\nprint(1)\n```", 0),
    ],
)
def test_withheld_fence_chars(text, expected):
    """_withheld_fence_chars withholds exactly an incomplete trailing fence.

    Pins the boundary the caller depends on: a trailing run of one or two
    backticks is held back because the next chunk may complete a fence, while
    three or more already form a fence and must be emitted immediately. The
    four-backtick case is the one that a "does it end with two backticks"
    implementation gets wrong, since it also ends with two.
    """
    assert _withheld_fence_chars(text) == expected


def test_mid_line_fence_does_not_close_code_block():
    """A ``` in the middle of a line is code content, not the closing fence.

    Models emit code that itself contains fences (a quoted ``` in a string
    literal, a heredoc writing Markdown). CommonMark only accepts a closing
    fence at the start of a line; anything else is body text that must reach
    the runner intact rather than truncating the block early.
    """
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\n"}}]},
            {"choices": [{"delta": {"content": 'x = "```"\nprint(1)\n'}}]},
            {"choices": [{"delta": {"content": "```"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "system"}]}))
    assert result == [
        {"type": "code", "format": "python", "content": 'x = "```"\nprint(1)\n'},
    ]


def test_indented_closing_fence_closes_the_block():
    """Up to three leading spaces are allowed on a closing fence."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\ncode\n"}}]},
            {"choices": [{"delta": {"content": "   ```\n"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert _code_of(result) == "code\n"


def test_closing_fence_with_trailing_text_is_code():
    """A line starting with ``` but trailing text does not close the block."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\ncode\n"}}]},
            {"choices": [{"delta": {"content": "```python\nmore\n"}}]},
            {"choices": [{"delta": {"content": "```\n"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert _code_of(result) == "code\n```python\nmore\n"


def test_split_closing_fence_candidate_is_held():
    """A fence split across deltas closes instead of being emitted as code."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\ncode\n  `"}}]},
            {"choices": [{"delta": {"content": "``\n"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert _code_of(result) == "code\n"


def test_split_candidate_with_trailing_text_is_code():
    """A held candidate that gains trailing text becomes ordinary code."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```python\ncode\n  ``"}}]},
            {"choices": [{"delta": {"content": "x\n"}}]},
            {"choices": [{"delta": {"content": "```\n"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert _code_of(result) == "code\n  ``x\n"


def test_unterminated_code_tail_that_is_not_a_closing_fence_is_emitted():
    """A cut-off code tail with ordinary text is emitted, not held as a fence."""
    llm = _make_llm([{"choices": [{"delta": {"content": "```python\nprint(1) x"}}]}])
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert _code_of(result) == "print(1) x"


def test_code_ending_at_a_newline_without_a_closing_fence_is_emitted():
    """Code ending on a newline is flushed when the stream ends without a fence."""
    llm = _make_llm([{"choices": [{"delta": {"content": "```python\nprint(1)\n"}}]}])
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert _code_of(result) == "print(1)\n"


def test_closing_fence_after_code_in_one_delta_is_emitted():
    """A complete block with a newline after the closing fence yields its code."""
    llm = _make_llm(
        [{"choices": [{"delta": {"content": "```python\nprint(1)\n```\n"}}]}]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert result == [{"type": "code", "format": "python", "content": "print(1)\n"}]


def test_empty_code_block_yields_nothing():
    """A fence that closes immediately has no code to emit."""
    llm = _make_llm([{"choices": [{"delta": {"content": "```python\n```\n"}}]}])
    assert list(run_text_llm(llm, {"messages": [{"content": "sys"}]})) == []


def test_four_tick_fence_keeps_inner_three_tick_line():
    """A closing run must be at least as long as the opening run."""
    llm = _make_llm(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "content": "````markdown\nfirst\n```\ninner-fence-is-code\n````\n"
                        }
                    }
                ]
            }
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert result == [
        {
            "type": "code",
            "format": "markdown",
            "content": "first\n```\ninner-fence-is-code\n",
        }
    ]


def test_four_tick_unlabelled_fence_is_not_lost():
    """The extra opening backtick is part of the fence, not the language label."""
    llm = _make_llm(
        [{"choices": [{"delta": {"content": "````\nprint(1)\n````\n"}}]}]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert result == [{"type": "code", "format": "python", "content": "print(1)\n"}]


def test_four_tick_opener_split_across_deltas():
    """A split extra backtick still counts toward the required closing length."""
    llm = _make_llm(
        [
            {"choices": [{"delta": {"content": "```"}}]},
            {"choices": [{"delta": {"content": "`python\nprint(1)\n"}}]},
            {"choices": [{"delta": {"content": "````\n"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "sys"}]}))
    assert result == [{"type": "code", "format": "python", "content": "print(1)\n"}]
