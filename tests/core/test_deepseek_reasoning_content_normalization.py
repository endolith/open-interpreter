import litellm

import interpreter.core.llm.llm as llm_mod

PLACEHOLDER = "."
LEGACY_PLACEHOLDER = "Executing the requested command."


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute",
            "description": "run code",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _process(messages, model="deepseek/deepseek-v4-flash", tools=_TOOLS, **extra):
    """Run messages through the outgoing-request normalization and return them.

    fixed_litellm_completions mutates the message list (filling reasoning_content
    for DeepSeek thinking mode) and then hands it to litellm.completion. Capture
    the list it actually sends by stubbing litellm.completion with an empty stream.

    `tools` defaults to a tool schema because the guide makes passback conditional
    on the request carrying tools; pass tools=None to model a request without them.
    """
    captured = {}

    def fake_completion(**params):
        captured["messages"] = params["messages"]
        return iter(())

    kwargs = {"model": model, "messages": messages, "stream": True, **extra}
    if tools is not None:
        kwargs["tools"] = tools

    original = litellm.completion
    litellm.completion = fake_completion
    try:
        list(llm_mod.fixed_litellm_completions(**kwargs))
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


def test_bare_wire_name_gets_fill():
    """The opencode_go route strips the prefix before this point, so the model
    arrives as the bare wire id with no prefix at all. That spelling must
    still match: it 400d in production (opencode_go/deepseek-v4.1-flash)
    because only prefixed spellings were matched.
    """
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    out = _process(messages, model="deepseek-v4.1-flash")
    assert out[1]["reasoning_content"] == PLACEHOLDER


def test_thinking_history_triggers_fill_for_unknown_model():
    """A request already carrying real reasoning is in thinking mode with its
    provider, whatever the model string says. A later reasoning-less
    tool-call turn must still be filled, even when the name matches nothing.
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
    out = _process(messages, model="opencode_go/some-thinking-model")
    assert out[3]["reasoning_content"] == "real chain of thought"


def test_no_thinking_history_leaves_unknown_model_untouched():
    """Without any reasoning in the request there is no evidence of thinking
    mode, so a non-matching model must go out byte-identical: adding the
    field speculatively could break providers that reject unknown fields.
    """
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    out = _process(messages, model="opencode_go/some-plain-model")
    assert "reasoning_content" not in out[1]
    assert "reasoning_content" not in out[3]


def test_without_tools_leaves_deepseek_untouched():
    """Without tools the field is ignored, so the request must stay byte-identical.

    The guide makes passback conditional on the request carrying tools; filling
    a text-only request only mutates stored messages and bloats the payload.
    """
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    out = _process(messages, tools=None)
    assert "reasoning_content" not in out[1]


def test_placeholder_only_history_does_not_trigger_fill():
    """Placeholders are ours, not the provider's: a history holding only "."
    (real reasoning trimmed away, or a mid-conversation switch to a
    non-thinking model) must not keep the fill running on speculation.
    """
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": PLACEHOLDER,
            "tool_calls": [_tool_call("c1")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c2")]},
    ]
    out = _process(messages, model="opencode_go/some-plain-model")
    assert "reasoning_content" not in out[3]


def test_legacy_placeholder_on_plain_assistant_becomes_empty():
    """A legacy placeholder on a non-tool turn is normalized to "".

    Only tool_calls messages need a non-empty value; a plain assistant turn
    accepts "", and forwarding the fabricated phrase would teach the model
    that narrating an action is acceptable.
    """
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "done",
            "reasoning_content": LEGACY_PLACEHOLDER,
        },
    ]
    out = _process(messages)
    assert out[1]["reasoning_content"] == ""


def test_empty_reasoning_on_tool_call_gets_placeholder():
    """A tool_calls message carrying "" is as fatal as a missing field."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [_tool_call("c1")],
        },
    ]
    out = _process(messages)
    assert out[1]["reasoning_content"] == PLACEHOLDER


def test_extra_body_disable_leaves_messages_untouched():
    """An explicit include_reasoning=False in extra_body disables the fill."""
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1")]},
    ]
    out = _process(messages, extra_body={"include_reasoning": False})
    assert "reasoning_content" not in out[1]


def test_top_level_disable_leaves_messages_untouched():
    """An explicit include_reasoning=False in params disables the fill."""
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1")]},
    ]
    out = _process(messages, include_reasoning=False)
    assert "reasoning_content" not in out[1]


def test_non_string_reasoning_is_not_thinking_history():
    """A non-string reasoning_content is not evidence of a thinking provider."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": None,
            "tool_calls": [_tool_call("c1")],
        },
    ]
    out = _process(messages, model="opencode_go/some-plain-model")
    assert out[1]["reasoning_content"] is None


def test_non_string_reasoning_on_tool_call_gets_placeholder():
    """A relay can send a list; it must be treated as no reasoning, not crash."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": ["not", "a", "string"],
            "tool_calls": [_tool_call("c1")],
        },
    ]
    out = _process(messages)
    assert out[1]["reasoning_content"] == PLACEHOLDER


def test_non_string_reasoning_on_plain_assistant_becomes_empty():
    """A non-string reasoning value on a plain turn is normalized to ""."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "done",
            "reasoning_content": {"unexpected": "shape"},
        },
    ]
    out = _process(messages)
    assert out[1]["reasoning_content"] == ""


def test_thinking_disabled_leaves_messages_untouched():
    """DeepSeek's own thinking-off switch disables the fill."""
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1")]},
    ]
    out = _process(messages, extra_body={"thinking": {"type": "disabled"}})
    assert "reasoning_content" not in out[1]
