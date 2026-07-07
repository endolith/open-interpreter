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
