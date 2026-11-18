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


def test_parallel_tool_calls_are_not_repaired_again():
    """Answers to two calls from one assistant turn must not each get a synthetic call.

    The orphan repair used to look only at the previous message, so the second
    answer (whose predecessor is the first answer, not the assistant) was treated
    as orphaned. That declared its call a second time and left the first call
    without an answer before the next assistant message, a sequence providers
    reject. Tracking the ids declared by the most recent assistant message keeps
    both answers paired without inserting anything.
    """
    messages = [
        {"role": "user", "content": "run both"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "execute", "arguments": "{}"},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "execute", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "a"},
        {"role": "tool", "tool_call_id": "call_b", "content": "b"},
    ]
    result = process_messages(messages)
    assert len(result) == 4
    assert [m["role"] for m in result] == ["user", "assistant", "tool", "tool"]
    assert result[1]["tool_calls"] == messages[1]["tool_calls"]
    assert result[2]["tool_call_id"] == "call_a"
    assert result[3]["tool_call_id"] == "call_b"


def test_unmatched_tool_result_between_answers_joins_the_declaring_turn():
    """An unmatchable answer between two real answers is declared by the same turn.

    A provider can reject one of several parallel calls and return a tool response
    for it. Inserting a synthetic assistant message for that answer puts it between
    the turn's remaining answers, so the later answer no longer follows the turn
    that declared its call and the provider rejects the history. The call belongs
    on the assistant message that declared the other calls of that turn.
    """
    messages = [
        {"role": "user", "content": "run both"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "execute", "arguments": "{}"},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "execute", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "a"},
        {"role": "tool", "tool_call_id": "call_unknown", "content": "provider error"},
        {"role": "tool", "tool_call_id": "call_b", "content": "b"},
    ]
    result = process_messages(messages)
    assert len(result) == 5
    assert [m["role"] for m in result] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
    ]
    assert [call["id"] for call in result[1]["tool_calls"]] == [
        "call_a",
        "call_b",
        "call_unknown",
    ]


def test_generated_tool_id_skips_ids_already_in_the_history():
    """A synthesized call id must not collide with an id the history already uses.

    An assistant turn can declare calls with provider-assigned ids without
    updating the legacy counter, so generating `toolu_1` for a tool message that
    lost its id would declare that call a second time. Reusing an id makes the
    identity of two calls ambiguous, so the allocator has to skip used ids.
    """
    messages = [
        {"role": "user", "content": "run both"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "execute", "arguments": "{}"},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "execute", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "a"},
        {"role": "tool", "content": "provider error"},
        {"role": "tool", "tool_call_id": "call_b", "content": "b"},
    ]
    result = process_messages(messages)
    assert len(result) == 5
    declared = [
        call["id"] for message in result for call in message.get("tool_calls") or []
    ]
    assert len(declared) == len(set(declared))
    generated = result[3]["tool_call_id"]
    assert generated not in {"toolu_1", "call_b"}
    assert generated in declared


def test_orphan_function_response_owns_the_next_unmatched_answer():
    """An answer after a synthesized orphan turn attaches to that turn.

    The orphan `function` branch emits its own assistant declaration. If that
    message is not recorded as the pending turn, the next undeclared tool result
    is attached to the turn before it, so its declaration ends up separated from
    its answer by the turn emitted in between.
    """
    messages = [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "execute", "arguments": "{}"},
        },
        {"role": "function", "name": "execute", "content": "first"},
        {"role": "function", "name": "execute", "content": "second"},
        {"role": "tool", "tool_call_id": "call_x", "content": "provider error"},
    ]
    result = process_messages(messages)
    assert len(result) == 6
    assert [m["role"] for m in result] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "tool",
    ]
    assert [call["id"] for call in result[3]["tool_calls"]] == ["toolu_2", "call_x"]
    assert result[4]["tool_call_id"] == "toolu_2"
    assert result[5]["tool_call_id"] == "call_x"
