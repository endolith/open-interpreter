from interpreter.core.llm.run_tool_calling_llm import process_messages


def test_function_call_converted_to_tool_calls():
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
    messages = [
        {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "execute", "arguments": "{}"},
        }
    ]
    result = process_messages(messages)
    assert len(result) == 2
    assert result[1] == {"role": "tool", "tool_call_id": "toolu_1", "content": ""}


def test_orphaned_function_response_gets_synthetic_tool_call():
    messages = [{"role": "function", "name": "execute", "content": "late output"}]
    result = process_messages(messages)
    assert result[0]["role"] == "assistant"
    assert result[1]["role"] == "tool"


def test_passthrough_message_unchanged():
    messages = [{"role": "user", "content": "hello"}]
    assert process_messages(messages) == messages
