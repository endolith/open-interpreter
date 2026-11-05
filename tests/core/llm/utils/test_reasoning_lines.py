from types import SimpleNamespace

from interpreter.core.llm.run_text_llm import run_text_llm
from interpreter.core.llm.run_tool_calling_llm import run_tool_calling_llm
from interpreter.core.llm.utils.convert_to_openai_messages import (
    convert_to_openai_messages,
)
from interpreter.core.llm.utils.reasoning_lines import (
    FillerLineFilter,
    is_filler_line,
    strip_filler_reasoning_lines,
)


def test_strip_removes_empty_directive_lines():
    """Whole reasoning lines that only say run/execute/let's do it are dropped."""
    text = "I should inspect the file.\nRun.\nExecute.\nLet's do it.\nIt may fail."
    assert strip_filler_reasoning_lines(text) == (
        "I should inspect the file.\nIt may fail."
    )


def test_strip_keeps_lines_with_actual_content():
    """A line that merely starts with a directive keeps its content and is not removed."""
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


def test_filter_streams_ordinary_reasoning_without_delay():
    """Reasoning that cannot become filler is emitted immediately, not buffered."""
    line_filter = FillerLineFilter()
    assert line_filter.feed("The file uses tabs") == "The file uses tabs"
    assert line_filter.feed(".\nNext") == ".\nNext"
    assert line_filter.flush() == ""


def test_filter_drops_filler_split_across_deltas():
    """A filler line arriving token-by-token is held and dropped once complete."""
    line_filter = FillerLineFilter()
    assert line_filter.feed("Run") == ""
    assert line_filter.feed(".") == ""
    assert line_filter.feed("\n") == ""
    assert line_filter.feed("Now check the log.\n") == "Now check the log.\n"


def test_filter_flushes_a_non_filler_remainder():
    """A still-ambiguous trailing line is emitted on flush when it is not filler."""
    line_filter = FillerLineFilter()
    assert line_filter.feed("Let's do") == ""
    assert line_filter.flush() == "Let's do"


def test_filter_drops_a_trailing_filler_without_newline():
    """A filler line at end of stream, without a trailing newline, is still dropped."""
    line_filter = FillerLineFilter()
    assert line_filter.feed("Execute") == ""
    assert line_filter.flush() == ""


def _text_llm(chunks):
    def completions(**params):
        yield from chunks

    return SimpleNamespace(
        completions=completions,
        execution_instructions=None,
        interpreter=SimpleNamespace(verbose=False, os=False),
    )


def test_run_text_llm_keeps_a_content_line_split_across_reasoning_deltas():
    """A reasoning line that starts like filler but continues with content is kept whole."""
    llm = _text_llm(
        [
            {"choices": [{"delta": {"reasoning_content": "Run"}}]},
            {"choices": [{"delta": {"reasoning_content": " the tests.\n"}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "system"}]}))
    reasoning = "".join(
        chunk["content"]
        for chunk in result
        if chunk.get("format") == "reasoning"
    )
    assert reasoning == "Run the tests.\n"


def test_run_text_llm_strips_filler_reasoning():
    """Text mode never shows filler-only reasoning lines to the user or stores them."""
    llm = _text_llm(
        [
            {"choices": [{"delta": {"content": None, "reasoning_content": "Check the file.\n"}}]},
            {"choices": [{"delta": {"content": None, "reasoning_content": "Run.\n"}}]},
            {"choices": [{"delta": {"content": "Here is the answer."}}]},
        ]
    )
    result = list(run_text_llm(llm, {"messages": [{"content": "system"}]}))
    assert result == [
        {
            "role": "assistant",
            "type": "message",
            "format": "reasoning",
            "content": "Check the file.\n",
        },
        {"type": "message", "content": "Here is the answer."},
    ]


class _Lang:
    name = "Python"


def _tool_llm(chunks):
    def completions(**params):
        yield from chunks

    return SimpleNamespace(
        completions=completions,
        model="test-model",
        tool_calling_instructions="",
        interpreter=SimpleNamespace(
            llm=SimpleNamespace(supports_vision=False, tool_calling_instructions=""),
            terminal=SimpleNamespace(languages=[_Lang()]),
            toolbox=SimpleNamespace(terminal=SimpleNamespace(languages=[_Lang()])),
            verbose=False,
            debug=False,
        ),
    )


def test_run_tool_calling_llm_replaces_filler_reasoning_with_clean_text():
    """The replace chunk that repaints reasoning carries no filler-only lines."""
    llm = _tool_llm(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "Inspect the file.\nLet's do it.\n"
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"content": "Done inspecting."}}]},
        ]
    )
    result = list(run_tool_calling_llm(llm, {"messages": []}))
    reasoning = "".join(
        chunk["content"]
        for chunk in result
        if chunk.get("format") == "reasoning" and not chunk.get("replace")
    )
    replaces = [
        chunk
        for chunk in result
        if chunk.get("format") == "reasoning" and chunk.get("replace")
    ]
    assert "Let's do it" not in reasoning
    assert replaces and replaces[0]["content"] == "Inspect the file.\n\n"


def test_converter_strips_stored_reasoning_before_sending_it_back():
    """Old saved filler reasoning is not echoed to the provider on later turns."""
    interpreter = SimpleNamespace(
        user_message_template="User: {content}",
        always_apply_user_message_template=False,
        code_output_template="Output:\n{content}",
        empty_code_output_template="(no output)",
        code_output_sender="user",
        debug=False,
    )
    messages = [
        {"role": "user", "type": "message", "content": "go"},
        {
            "role": "assistant",
            "type": "message",
            "content": "hi",
            "reasoning_content": "Plan the change.\nRun.\nLet's do it.",
        },
    ]
    result = convert_to_openai_messages(messages, interpreter=interpreter)
    assert result[-1]["reasoning_content"] == "Plan the change.\n"
