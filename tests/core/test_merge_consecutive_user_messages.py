from interpreter.core.llm.run_tool_calling_llm import (
    merge_consecutive_user_messages,
    process_messages,
)


def test_merges_consecutive_user_strings():
    """Adjacent user messages collapse into one, joined by a blank line.

    DeepSeek's chat format requires interleaved user/assistant messages
    (deepseek-ai/DeepSeek-R1#21). OI builds runs of user messages when resuming
    a session and attaching images, so they must be merged before sending.
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
    """Text plus image parts merge into one multimodal user message in order.

    Attaching an image appends the path as a text message and then the image
    itself; merging must keep both and preserve their order rather than drop
    the image or the text.
    """
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
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
    """Only truly adjacent user messages merge; assistant/tool messages break the run.

    The API requires the assistant message and its tool results to stay between
    user turns, so merging across them would corrupt the tool-call pairing.
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
    text, then the image itself. After processing there must be no two adjacent
    user messages, or DeepSeek's template misbehaves.
    """
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "SYSTEM ALERT: This conversation was resumed."},
        {"role": "user", "content": "D:\\path\\_tmp_r77.png"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "[2026-09-10 15:22] "}, image_part],
        },
    ]
    processed = process_messages(messages, model="deepseek/deepseek-v4-flash")

    roles = [m.get("role") for m in processed]
    for a, b in zip(roles, roles[1:]):
        assert not (a == "user" and b == "user"), roles

    last = processed[-1]
    assert last["role"] == "user"
    assert isinstance(last["content"], list)
    assert last["content"][-1] == image_part
