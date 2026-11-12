from interpreter.core.llm.run_tool_calling_llm import (
    process_messages,
    tool_conversion_overhead,
)


def test_function_call_converted_to_tool_calls():
    """Legacy function_call/function messages are rewritten as assistant tool_calls plus tool responses."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "execute", "arguments": "{}"},
        },
        {"role": "function", "name": "execute", "content": "output"},
    ]
    result = process_messages(messages)
    assert result[0]["tool_calls"][0]["id"] == "toolu_1"
    assert result[1]["role"] == "tool"
    assert result[1]["tool_call_id"] == "toolu_1"


def test_function_call_without_response_gets_empty_tool():
    """An assistant function_call with no following function message gets a synthetic empty tool reply."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "execute", "arguments": "{}"},
        }
    ]
    result = process_messages(messages)
    assert len(result) == 2
    assert result[1] == {"role": "tool",
                         "tool_call_id": "toolu_1",
                         "content": ""}


def test_orphaned_function_response_gets_synthetic_tool_call():
    """A lone function message is paired with a synthetic assistant tool_call so the API sees a valid pair."""
    messages = [{"role": "function",
                 "name": "execute",
                 "content": "late output"}]
    result = process_messages(messages)
    assert result[0]["role"] == "assistant"
    assert result[0]["tool_calls"][0]["id"] == "toolu_1"
    assert result[1] == {
        "role": "tool",
        "name": "execute",
        "content": "late output",
        "tool_call_id": "toolu_1",
    }


def test_passthrough_message_unchanged():
    """Messages that are already in tool-calling format are returned without modification."""
    messages = [{"role": "user", "content": "hello"}]
    assert process_messages(messages) == messages


def test_process_messages_empty_list():
    """An empty message list is returned unchanged."""
    assert process_messages([]) == []


def test_process_messages_mutates_function_call_in_place():
    """function_call is popped and replaced with tool_calls on the original message dict."""
    message = {
        "role": "assistant",
        "content": "",
        "function_call": {"name": "execute", "arguments": "{}"},
    }
    process_messages([message])
    assert "function_call" not in message
    assert message["tool_calls"][0]["id"] == "toolu_1"


def test_sequential_function_calls_get_incrementing_tool_ids():
    """Each function_call/function pair receives a new toolu_N id in order."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "execute", "arguments": "{}"},
        },
        {"role": "function", "name": "execute", "content": "first"},
        {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "execute", "arguments": "{}"},
        },
        {"role": "function", "name": "execute", "content": "second"},
    ]
    result = process_messages(messages)
    assert result[0]["tool_calls"][0]["id"] == "toolu_1"
    assert result[1]["tool_call_id"] == "toolu_1"
    assert result[2]["tool_calls"][0]["id"] == "toolu_2"
    assert result[3]["tool_call_id"] == "toolu_2"


def test_tool_conversion_overhead_counts_the_rewrite_delta():
    """The trim budget must reserve the tokens the runner adds when it converts
    legacy function_call turns into modern tool_calls: generated ids, type
    markers, tool_call_id fields and any synthetic tool response."""
    messages = [
        {"role": "user", "content": "run this"},
        {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "execute", "arguments": '{"code": "1"}'},
        },
        {"role": "function", "name": "execute", "content": "1"},
    ]
    assert tool_conversion_overhead(messages, "gpt-4o") > 0


def test_tool_conversion_overhead_is_zero_for_modern_messages():
    """A history already in tool_calls form is passed through unchanged, so the
    request costs the same before and after the runner's conversion."""
    messages = [
        {"role": "user", "content": "run this"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "execute",
                        "arguments": '{"code": "1"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "1"},
    ]
    assert tool_conversion_overhead(messages, "gpt-4o") == 0


def test_tool_conversion_overhead_does_not_mutate_the_history():
    """The delta is measured on a copy; measuring must not pop function_call or
    rewrite roles on the messages the caller is about to trim and send."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "execute", "arguments": "{}"},
        }
    ]
    tool_conversion_overhead(messages, None)
    assert messages[0]["function_call"] == {"name": "execute", "arguments": "{}"}
    assert "tool_calls" not in messages[0]


def test_tool_conversion_overhead_grows_with_converted_turns():
    """Each converted turn adds its own id and wrappers, so a longer retained
    history needs a larger reserve than a single turn."""
    def history(pairs):
        messages = []
        for _ in range(pairs):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {"name": "execute", "arguments": "{}"},
                }
            )
            messages.append({"role": "function", "name": "execute", "content": "x"})
        return messages

    one = tool_conversion_overhead(history(1), None)
    three = tool_conversion_overhead(history(3), None)
    assert one > 0
    assert three > one
