import litellm

import interpreter.core.llm.llm as llm_mod

PLACEHOLDER = "."
LEGACY_PLACEHOLDER = "Executing the requested command."


def _process(messages, model="deepseek/deepseek-v4-flash"):
    """Run messages through the outgoing-request normalization and return them.

    fixed_litellm_completions mutates the message list (filling reasoning_content
    for DeepSeek thinking mode) and then hands it to litellm.completion. Capture
    the list it actually sends by stubbing litellm.completion with an empty stream.
    """
    captured = {}

    def fake_completion(**params):
        captured["messages"] = params["messages"]
        return iter(())

    original = litellm.completion
    litellm.completion = fake_completion
    try:
        list(
            llm_mod.fixed_litellm_completions(
                model=model,
                messages=messages,
                stream=True,
            )
        )
    finally:
        litellm.completion = original
    return captured["messages"]


def _tool_call(call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "execute", "arguments": "{}"},
    }


def test_real_reasoning_is_preserved_never_replaced_by_placeholder():
    """A tool-call turn that actually reasoned keeps its reasoning verbatim.

    The placeholder exists only for turns with no reasoning; substituting it
    where the model did think would feed fabricated thoughts back as context.
    """
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should list the directory first.",
            "tool_calls": [_tool_call("c1")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    out = _process(messages)
    assert out[1]["reasoning_content"] == "I should list the directory first."


def test_missing_reasoning_on_tool_call_turn_gets_placeholder():
    """A tool-call turn with no reasoning gets the non-empty placeholder.

    DeepSeek/OpenRouter reject a tool_calls message whose reasoning_content is
    missing or empty, so a non-empty filler is required there.
    """
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    out = _process(messages)
    assert out[1]["reasoning_content"] == PLACEHOLDER


def test_missing_reasoning_on_plain_assistant_turn_is_empty_string():
    """A non-tool assistant turn with no reasoning uses "" (empty is accepted there).

    Only tool_calls turns must be non-empty; plain assistant turns may carry "".
    """
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "done"},
    ]
    out = _process(messages)
    assert out[1]["reasoning_content"] == ""


def test_legacy_placeholder_is_treated_as_no_reasoning():
    """The old "Executing the requested command." marker is not forwarded as reasoning.

    Conversations stored by older versions contain that fabricated phrase; treating
    it as real reasoning would teach the model that narrating actions is acceptable,
    so it is normalized away like any other empty turn.
    """
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": LEGACY_PLACEHOLDER,
            "tool_calls": [_tool_call("c1")],
        },
    ]
    out = _process(messages)
    assert out[1]["reasoning_content"] == PLACEHOLDER


def test_turn_reasoning_propagates_to_later_tool_calls_without_reasoning():
    """One turn's real reasoning fills later tool-call messages in the same turn.

    A multi-tool turn is stored as several assistant tool-call messages separated
    by tool results; each must carry the turn's reasoning or DeepSeek 400s.
    """
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "real chain of thought",
            "tool_calls": [_tool_call("c1")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c2")]},
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
    ]
    out = _process(messages)
    assert out[1]["reasoning_content"] == "real chain of thought"
    assert out[3]["reasoning_content"] == "real chain of thought"
