import json

from interpreter.core.llm.run_tool_calling_llm import process_messages


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


def test_orphaned_tool_message_gets_synthetic_assistant_tool_call():
    """A tool message with no preceding assistant tool_calls is paired with a synthetic one.

    Providers reject a role "tool" message whose tool_call_id has no matching
    assistant tool_calls entry (the error text would otherwise be followed by a
    second rejection on the next request). The synthetic call reuses the existing
    tool_call_id so the pair stays matched, and its arguments are valid JSON so a
    strict provider does not reject the placeholder itself.
    """
    messages = [
        {"role": "user", "content": "run this"},
        {"role": "assistant", "content": "I'll try."},
        {"role": "tool", "tool_call_id": "call_abc", "content": "Invalid tool call"},
    ]
    result = process_messages(messages)
    assert result[0]["role"] == "user"
    assert result[1] == {"role": "assistant", "content": "I'll try."}
    assert result[2]["role"] == "assistant"
    assert result[2]["tool_calls"][0]["id"] == "call_abc"
    # The placeholder body must round-trip as JSON; providers validate arguments.
    assert json.loads(result[2]["tool_calls"][0]["function"]["arguments"])["language"] == "python"
    assert result[3]["role"] == "tool"
    assert result[3]["tool_call_id"] == "call_abc"
    assert result[3]["content"] == "Invalid tool call"


def test_orphaned_tool_message_without_id_gets_generated_id():
    """A tool message without a tool_call_id gets a generated id shared with its synthetic call."""
    messages = [{"role": "user", "content": "run this"}, {"role": "tool", "content": "error"}]
    result = process_messages(messages)
    assert result[1]["role"] == "assistant"
    generated = result[1]["tool_calls"][0]["id"]
    assert generated.startswith("toolu_")
    assert result[2]["role"] == "tool"
    assert result[2]["tool_call_id"] == generated


def test_tool_message_after_assistant_tool_calls_is_not_repaired():
    """An already-paired tool message is passed through without an extra synthetic assistant."""
    messages = [
        {"role": "user", "content": "run this"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "execute", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "output"},
    ]
    result = process_messages(messages)
    assert len(result) == 3
    assert result[1]["tool_calls"][0]["id"] == "call_abc"
    assert result[2] == messages[2]
