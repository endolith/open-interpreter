from interpreter.core.llm.run_tool_calling_llm import (
    merge_consecutive_assistant_messages,
    process_messages,
)
from interpreter.core.llm.utils.convert_to_openai_messages import (
    convert_to_openai_messages,
)


class _FakeInterpreter:
    """Minimal stand-in for the interpreter attributes convert_to_openai_messages reads."""

    always_apply_user_message_template = False
    user_message_template = "{content}"
    code_output_sender = "function"
    empty_code_output_template = ""
    debug = False


def test_merges_preamble_and_tool_call_into_single_assistant_turn():
    """A narration message followed by a tool_calls message becomes one assistant turn.

    Open Interpreter stores a tool-calling turn as a content-only preamble and a
    separate tool_calls message with empty content. DeepSeek V4's chat template
    renders the second consecutive assistant with orphaned thinking and no
    <|Assistant|> transition, which makes the model spill repeated inner-monologue
    into the content channel instead of calling the tool. The canonical OpenAI
    shape is a single assistant message carrying both content and tool_calls.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "Let me check the file.",
            "reasoning_content": "I should read the file.",
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should read the file.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "execute", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
    ]

    out = merge_consecutive_assistant_messages(messages)

    roles = [m["role"] for m in out]
    assert roles == ["system", "user", "assistant", "tool"]
    assert out[2]["content"] == "Let me check the file."
    assert len(out[2]["tool_calls"]) == 1
    assert out[2]["tool_calls"][0]["id"] == "call_1"


def test_dedupes_identical_duplicated_reasoning():
    """The reasoning copied onto both halves of a split turn is emitted once.

    Open Interpreter's reasoning propagation copies the same reasoning_content
    onto the preamble and the tool_calls message so DeepSeek accepts either one.
    After merging, echoing that block twice would bloat the prompt and reinforce
    the model's own degenerating repetition, so identical adjacent reasoning is
    collapsed to a single copy.
    """
    messages = [
        {"role": "assistant", "content": "hi", "reasoning_content": "same"},
        {"role": "assistant", "content": "", "reasoning_content": "same", "tool_calls": []},
    ]

    out = merge_consecutive_assistant_messages(messages)

    assert len(out) == 1
    assert out[0]["reasoning_content"] == "same"


def test_concatenates_distinct_reasoning_in_order():
    """Distinct reasoning blocks from the same turn are joined, preserving order.

    A turn can accumulate more than one reasoning block (multiple model calls).
    Merging must not drop any of them; it joins non-empty parts with a blank line
    so no reasoning is lost.
    """
    messages = [
        {"role": "assistant", "content": "first", "reasoning_content": "thought A"},
        {"role": "assistant", "content": "second", "reasoning_content": "thought B"},
    ]

    out = merge_consecutive_assistant_messages(messages)

    assert len(out) == 1
    assert out[0]["content"] == "first\n\nsecond"
    assert out[0]["reasoning_content"] == "thought A\n\nthought B"


def test_does_not_merge_across_tool_or_user_messages():
    """Only true runs of consecutive assistants merge; other roles are boundaries.

    Tool results and user turns mark real turn boundaries (and tool results must
    stay immediately after their tool_calls). Merging across them would produce
    invalid pairing, so they must block the merge.
    """
    messages = [
        {"role": "assistant", "content": "a"},
        {"role": "tool", "tool_call_id": "x", "content": "out"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "c"},
    ]

    out = merge_consecutive_assistant_messages(messages)

    assert [m["role"] for m in out] == [
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]


def test_concatenates_tool_calls_from_both_messages():
    """Tool calls split across consecutive assistant messages are concatenated.

    Parallel/continued tool calls can land in separate assistant messages; the
    merged turn must keep all of them so every tool response still has its
    matching call id.
    """
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b"}]},
    ]

    out = merge_consecutive_assistant_messages(messages)

    assert len(out) == 1
    assert [tc["id"] for tc in out[0]["tool_calls"]] == ["a", "b"]


def test_process_messages_produces_canonical_tool_turns():
    """End-to-end: process_messages emits no consecutive assistant messages.

    This is the property that matters for DeepSeek V4: the wire payload must not
    contain two assistant messages in a row, and each tool_calls message must be
    immediately followed by its tool response.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "Reading the file.",
            "reasoning_content": "plan",
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "plan",
            "function_call": {"name": "execute", "arguments": "{}"},
        },
        {"role": "function", "content": "output"},
    ]

    out = process_messages(messages, model="deepseek/deepseek-v4-flash")

    roles = [m["role"] for m in out]
    for a, b in zip(roles, roles[1:]):
        assert not (a == "assistant" and b == "assistant")

    assistant_with_call = next(m for m in out if m.get("tool_calls"))
    assert assistant_with_call["content"] == "Reading the file."
    assert assistant_with_call["reasoning_content"] == "plan"
    idx = out.index(assistant_with_call)
    assert out[idx + 1]["role"] == "tool"
    assert out[idx + 1]["tool_call_id"] == assistant_with_call["tool_calls"][0]["id"]


def test_full_pipeline_from_lmc_has_no_consecutive_assistants():
    """Regression: the real LMC -> API pipeline must never emit split assistant turns.

    This reproduces, deterministically and offline, the precondition that makes
    DeepSeek V4 spill repeated inner-monologue ("Let me do X ... Go ... Write")
    into the content channel: Open Interpreter stores a tool turn as a separate
    reasoning message, a narration message, and a code message. Conversion turns
    that into a content-only assistant message followed by a consecutive
    tool_calls assistant message that duplicates the same reasoning_content.
    DeepSeek's chat template has no separator for two consecutive assistants, so
    the second renders with orphaned thinking and a missing <|Assistant|> marker,
    and the corruption accumulates every tool round.

    Asserting the final wire payload has no consecutive assistant messages, no
    duplicated reasoning, and intact tool pairing locks the fix in place. The
    model-level loop itself is non-deterministic and cannot be asserted here.
    """
    lmc_messages = [
        {"role": "user", "type": "message", "content": "do the task"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "list the files\n\n"},
        {"role": "assistant", "type": "message", "content": "Listing files."},
        {"role": "assistant", "type": "code", "format": "python", "content": "import os; os.listdir('.')"},
        {"role": "computer", "type": "console", "format": "output", "content": "['a.txt']"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "read a.txt\n\n"},
        {"role": "assistant", "type": "message", "content": "Reading a.txt."},
        {"role": "assistant", "type": "code", "format": "python", "content": "print(open('a.txt').read())"},
        {"role": "computer", "type": "console", "format": "output", "content": "hello"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "done\n\n"},
        {"role": "assistant", "type": "message", "content": "All done."},
    ]

    converted = convert_to_openai_messages(
        lmc_messages,
        function_calling=True,
        vision=False,
        shrink_images=False,
        interpreter=_FakeInterpreter(),
    )

    # The bug this guards against: conversion splits each tool turn into two
    # consecutive assistant messages that carry the same reasoning.
    converted_roles = [m["role"] for m in converted]
    assert any(a == "assistant" and b == "assistant" for a, b in zip(converted_roles, converted_roles[1:]))

    final = process_messages([dict(m) for m in converted], model="deepseek/deepseek-v4-flash")

    # 1. No two assistant messages in a row anywhere on the wire.
    final_roles = [m["role"] for m in final]
    assert not any(a == "assistant" and b == "assistant" for a, b in zip(final_roles, final_roles[1:]))

    # 2. Each assistant turn carries its reasoning exactly once (no duplication).
    for m in final:
        if m["role"] == "assistant" and m.get("reasoning_content"):
            assert m["reasoning_content"].count("list the files") <= 1
            assert m["reasoning_content"].count("read a.txt") <= 1

    # 3. Every tool_calls message is immediately followed by its matching tool result.
    for i, m in enumerate(final):
        if m.get("tool_calls"):
            assert final[i + 1]["role"] == "tool"
            assert final[i + 1]["tool_call_id"] == m["tool_calls"][0]["id"]

