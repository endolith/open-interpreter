from types import SimpleNamespace

from interpreter.core.llm.run_text_llm import run_text_llm


def _make_llm(chunks, execution_instructions=None):
    def completions(**params):
        for chunk in chunks:
            yield chunk

    return SimpleNamespace(
        completions=completions,
        execution_instructions=execution_instructions,
        interpreter=SimpleNamespace(verbose=False, os=False),
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
