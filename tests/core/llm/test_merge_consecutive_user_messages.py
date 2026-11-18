import pytest

from interpreter.core.llm.run_tool_calling_llm import (
    merge_consecutive_user_messages,
    process_messages,
)


def test_merges_consecutive_user_strings():
    """Adjacent user messages collapse into one, joined by a blank line.

    DeepSeek requires interleaved user/assistant messages
    (deepseek-ai/DeepSeek-R1#21), while OI builds runs of user messages when
    resuming a session or attaching an image. They must be merged before the
    request is sent.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "SYSTEM ALERT: resumed"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "ok"},
    ]
    merged = merge_consecutive_user_messages(messages)

    assert [m["role"] for m in merged] == ["system", "user", "assistant"]
    assert merged[1]["content"] == "SYSTEM ALERT: resumed\n\ncontinue"


def test_merges_multimodal_content_preserving_order():
    """Text plus image parts merge into one multimodal message in order.

    Attaching an image appends the path as a text message and the image itself;
    merging must keep both, in order, instead of dropping the image or text.
    """
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }
    messages = [
        {"role": "user", "content": "here is the plot"},
        {"role": "user", "content": [{"type": "text", "text": "plot.png"}, image_part]},
    ]
    merged = merge_consecutive_user_messages(messages)

    assert len(merged) == 1
    assert merged[0]["content"] == [
        {"type": "text", "text": "here is the plot"},
        {"type": "text", "text": "plot.png"},
        image_part,
    ]


def test_does_not_merge_users_separated_by_other_roles():
    """Only truly adjacent user messages merge; other roles break the run.

    Assistant messages and their tool results must stay between user turns, so
    merging across them would corrupt the tool-call pairing.
    """
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "tool", "tool_call_id": "1", "content": "d"},
        {"role": "user", "content": "e"},
        {"role": "user", "content": "f"},
    ]
    merged = merge_consecutive_user_messages(messages)

    assert [m["role"] for m in merged] == [
        "user",
        "assistant",
        "user",
        "tool",
        "user",
    ]
    assert merged[-1]["content"] == "e\n\nf"


def test_process_messages_leaves_no_consecutive_users():
    """process_messages canonicalizes the resume+image user run before sending.

    This is the real-world shape: a SYSTEM ALERT user message, the image path as
    text, then the image part. After processing there must be no two adjacent
    user messages or DeepSeek's chat template rejects the request.
    """
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "SYSTEM ALERT: This conversation was resumed."},
        {"role": "user", "content": "plot.png"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "[2026-09-10 15:22] "}, image_part],
        },
    ]
    processed = process_messages([dict(m) for m in messages])

    roles = [m["role"] for m in processed]
    for first, second in zip(roles, roles[1:]):
        assert not (first == "user" and second == "user"), roles

    assert processed[-1]["role"] == "user"
    assert isinstance(processed[-1]["content"], list)
    assert processed[-1]["content"][-1] == image_part


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("", "hi", "hi"),
        ("a", "   ", "a"),
    ],
    ids=["empty-first", "blank-second"],
)
def test_merges_strings_ignoring_blank_halves(first, second, expected):
    """Empty or whitespace-only halves are dropped instead of joined.

    Resuming a session can append a blank SYSTEM ALERT user message; joining it
    must not leave stray blank lines in the merged turn.
    """
    merged = merge_consecutive_user_messages([{"role": "user", "content": first}, {"role": "user", "content": second}])

    assert len(merged) == 1
    assert merged[0]["content"] == expected


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (None, "text", [{"type": "text", "text": "text"}]),
        ("text", None, [{"type": "text", "text": "text"}]),
        (None, [{"type": "text", "text": "part"}], [{"type": "text", "text": "part"}]),
        ([{"type": "text", "text": "part"}], None, [{"type": "text", "text": "part"}]),
    ],
    ids=["none-then-text", "text-then-none", "none-then-parts", "parts-then-none"],
)
def test_merges_missing_content_without_dropping_the_other_side(first, second, expected):
    """A user message with ``content=None`` merges without losing its neighbor.

    Some providers send a user message whose content is null; merging it with a
    text or multimodal neighbor must keep the neighbor instead of raising or
    emitting an empty message.
    """
    merged = merge_consecutive_user_messages([{"role": "user", "content": first}, {"role": "user", "content": second}])

    assert len(merged) == 1
    assert merged[0]["content"] == expected


def test_merge_of_empty_message_list_is_empty():
    """Merging no messages returns no messages.

    ``process_messages`` can receive an empty request; the merge pass must stay
    a no-op rather than raising on the empty list.
    """
    assert merge_consecutive_user_messages([]) == []
