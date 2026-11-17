from types import SimpleNamespace

import pytest

from interpreter.core.llm.utils.convert_to_openai_messages import (
    convert_to_openai_messages,
)
from interpreter.core.llm.utils.reasoning_lines import (
    is_filler_line,
    strip_filler_reasoning_lines,
)


@pytest.fixture
def interpreter():
    return SimpleNamespace(
        user_message_template="User: {content}",
        always_apply_user_message_template=False,
        code_output_template="Output:\n{content}",
        empty_code_output_template="(no output)",
        code_output_sender="user",
        debug=False,
    )


def test_strip_removes_empty_directive_lines():
    """Whole reasoning lines that only say run/execute/let's do it are dropped."""
    text = "I should inspect the file.\nRun.\nExecute.\nLet's do it.\nIt may fail."
    assert strip_filler_reasoning_lines(text) == (
        "I should inspect the file.\nIt may fail."
    )


def test_strip_keeps_lines_with_actual_content():
    """A line that merely starts with a directive keeps its content."""
    text = "Run the tests after the edit.\nExecute the plan carefully.\n"
    assert strip_filler_reasoning_lines(text) == text


def test_strip_ignores_case_punctuation_and_markdown():
    """Case, trailing punctuation, emphasis, bullets, and curly apostrophes do not hide filler."""
    text = (
        "- RUN!\n"
        "1. Execute.\n"
        "> **Let\u2019s do it.**\n"
        "*Okay*\n"
        "Real thought.\n"
    )
    assert strip_filler_reasoning_lines(text) == "Real thought.\n"


def test_strip_preserves_unchanged_text_identity():
    """Text with no filler lines is returned unchanged, so callers can avoid needless copies."""
    text = "A thought that stays.\nAnother thought.\n"
    assert strip_filler_reasoning_lines(text) is text


def test_is_filler_line_requires_a_whole_line_match():
    """Only lines that are entirely a directive count as filler."""
    assert is_filler_line(" Run. ")
    assert is_filler_line("let's do it!")
    assert not is_filler_line("Run it after lunch.")
    assert not is_filler_line("Done.")


def test_convert_strips_filler_from_replayed_message_reasoning(interpreter):
    """A stored assistant message replays real reasoning without filler directives."""
    messages = [
        {
            "role": "assistant",
            "type": "message",
            "content": "Done.",
            "reasoning_content": "Check the file.\nRun.\nExecute.\nReal thought.",
        }
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result[0]["reasoning_content"] == "Check the file.\nReal thought."


def test_convert_strips_filler_from_replayed_code_reasoning(interpreter):
    """Reasoning stored on an assistant code message is replayed without filler."""
    messages = [
        {
            "role": "assistant",
            "type": "code",
            "format": "python",
            "content": "print(1)",
            "reasoning_content": "Run.\nNow write the code.",
        }
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result[0]["reasoning_content"] == "Now write the code."


def test_convert_strips_filler_from_merged_replay(interpreter):
    """The non-function-calling merge path also replays only real reasoning."""
    messages = [
        {
            "role": "assistant",
            "type": "message",
            "content": "Done.",
            "reasoning_content": "Check the file.\nRun.\nReal thought.",
        },
        {
            "role": "assistant",
            "type": "code",
            "format": "python",
            "content": "print(1)",
            "reasoning_content": "Execute.\nNow write the code.",
        },
    ]
    result = convert_to_openai_messages(
        messages, function_calling=False, interpreter=interpreter
    )
    assert result[0]["reasoning_content"] == (
        "Check the file.\nReal thought.\nNow write the code."
    )
