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
