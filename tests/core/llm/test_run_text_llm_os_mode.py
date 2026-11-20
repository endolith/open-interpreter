"""Regression tests for #392: unlabelled fences in OS mode.

The default-language branch tested `os == False` twice, so with
os=True neither fired, language stayed empty, and the block was
silently dropped (nothing yielded at all). OS mode now labels such
blocks "text"; normal mode still defaults to "python".
"""

from types import SimpleNamespace

from interpreter.core.llm.run_text_llm import run_text_llm


def _fake_llm(os_mode, content):
    def completions(**params):
        yield {"choices": [{"delta": {"content": content}}]}

    return SimpleNamespace(
        execution_instructions="",
        interpreter=SimpleNamespace(os=os_mode, verbose=False),
        completions=completions,
    )


def _code_chunks(llm):
    return [
        chunk
        for chunk in run_text_llm(llm, {"messages": []})
        if chunk.get("type") == "code"
    ]


def test_unlabelled_fence_in_os_mode_yields_text():
    """An unlabelled fence reaches the terminal as text in OS mode."""
    llm = _fake_llm(True, "```\nmeeting notes\n```\n")

    chunks = _code_chunks(llm)

    assert chunks, "OS mode dropped the unlabelled block entirely"
    assert chunks[0]["format"] == "text"


def test_unlabelled_fence_defaults_to_python():
    """Normal mode behavior is unchanged: unlabelled means python."""
    llm = _fake_llm(False, "```\nprint(1)\n```\n")

    chunks = _code_chunks(llm)

    assert chunks
    assert chunks[0]["format"] == "python"
