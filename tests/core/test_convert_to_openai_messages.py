from interpreter.core.llm.utils.convert_to_openai_messages import convert_to_openai_messages


class _FakeInterpreter:
    """Minimal stand-in for the interpreter attributes convert_to_openai_messages touches."""

    always_apply_user_message_template = False
    user_message_template = "{content}"
    code_output_sender = "function"
    empty_code_output_template = ""
    debug = False


def test_computer_role_description_image_becomes_user():
    messages = [
        {
            "role": "computer",
            "type": "image",
            "format": "description",
            "content": "A screenshot of bold text.",
            "recipient": "assistant",
        }
    ]
    out = convert_to_openai_messages(
        messages, function_calling=True, vision=False, interpreter=None
    )
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert "screenshot" in out[0]["content"]


def test_html_skips_png_when_not_vision():
    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    interpreter.llm.supports_vision = False
    html = interpreter.terminal.get_language("html")
    html_lang = html(interpreter)
    chunks = list(html_lang.run("<b>edited</b>"))
    image_chunks = [c for c in chunks if c.get("type") == "image"]
    assert image_chunks == []
    assistant_chunks = [
        c
        for c in chunks
        if c.get("recipient") == "assistant" and c.get("type") == "console"
    ]
    assert any("```html" in c.get("content", "") for c in assistant_chunks)


def test_react_incompatible_code_reports_error_without_crashing():
    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    react = interpreter.terminal.get_language("react")
    react_lang = react(interpreter)
    chunks = list(react_lang.run("import React from 'react'"))
    assert chunks
    assert chunks[0]["type"] == "console"
    assert "React format not supported" in chunks[0]["content"]
    assert "require" in chunks[0]["content"] or "import" in chunks[0]["content"]


def test_react_compatible_code_yields_html_for_user():
    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    interpreter.llm.supports_vision = False
    react = interpreter.terminal.get_language("react")
    react_lang = react(interpreter)
    code = 'ReactDOM.render(<h1>Hello, edit!</h1>, document.getElementById("root"));'
    chunks = list(react_lang.run(code))
    user_html = [c for c in chunks if c.get("recipient") == "user" and c.get("format") == "html"]
    assert len(user_html) == 1
    assert "Hello, edit!" in user_html[0]["content"]
    assert "type=\"text/babel\"" in user_html[0]["content"]


def test_reasoning_content_propagates_to_all_tool_calls_in_multi_code_turn():
    """Every tool-call message in a single turn must carry the turn's reasoning_content.

    DeepSeek's thinking mode returns a 400 if any assistant tool-call message in the
    history lacks reasoning_content. When one response produces several code blocks
    separated by tool output, the old code dropped the pending reasoning at the first
    tool/function message, leaving later tool calls unadorned. This verifies the
    reasoning survives intervening tool responses so no tool-call message is bare.
    """
    messages = [
        {"role": "user", "type": "message", "content": "do A and B"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "Plan. \n\n"},
        {"role": "assistant", "type": "message", "content": "Doing A."},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('B')"},
        {"role": "computer", "type": "console", "format": "output", "content": "B"},
    ]
    out = convert_to_openai_messages(
        messages, function_calling=True, vision=False, interpreter=_FakeInterpreter()
    )
    function_calls = [m for m in out if "function_call" in m]
    assert len(function_calls) == 2
    assert all(m["reasoning_content"] == "Plan. \n\n" for m in function_calls)


def test_tool_loop_reasoning_replaced_per_llm_call():
    """A new reasoning block after tool output replaces, not appends to, the old one.

    The tool loop makes a fresh LLM call per tool, so each call's reasoning is stored as
    its own reasoning message separated by tool output. The pending reasoning must reset
    for the new call (otherwise the second tool-call message would receive the first
    call's reasoning plus the second's, polluting history with stale thoughts).
    """
    messages = [
        {"role": "user", "type": "message", "content": "do A then B"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "R1. \n\n"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
        {"role": "computer", "type": "console", "format": "output", "content": "A"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "R2. \n\n"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('B')"},
        {"role": "computer", "type": "console", "format": "output", "content": "B"},
    ]
    out = convert_to_openai_messages(
        messages, function_calling=True, vision=False, interpreter=_FakeInterpreter()
    )
    function_calls = [m for m in out if "function_call" in m]
    assert len(function_calls) == 2
    assert function_calls[0]["reasoning_content"] == "R1. \n\n"
    assert function_calls[1]["reasoning_content"] == "R2. \n\n"


def test_post_stream_reasoning_backfills_earlier_assistant_messages():
    """Reasoning stored after the code it belongs to must still reach that tool-call message.

    Some providers deliver reasoning_content only once the stream completes, and the
    reasoning message can end up stored after the content/code messages in history.
    Without backfilling, the tool-call message would be sent without reasoning_content
    and DeepSeek would reject the request with a 400.
    """
    messages = [
        {"role": "user", "type": "message", "content": "compute"},
        {"role": "assistant", "type": "message", "content": "Running now."},
        {"role": "assistant", "type": "code", "format": "python", "content": "print(2+2)"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "Computed 2+2. \n\n"},
        {"role": "computer", "type": "console", "format": "output", "content": "4"},
    ]
    out = convert_to_openai_messages(
        messages, function_calling=True, vision=False, interpreter=_FakeInterpreter()
    )
    function_calls = [m for m in out if "function_call" in m]
    assert len(function_calls) == 1
    assert function_calls[0]["reasoning_content"] == "Computed 2+2. \n\n"


def test_whitespace_only_assistant_separator_is_dropped():
    """A stored loop-mode separator must never reach the API as an empty assistant message.

    respond() stores an assistant message containing only "\\n\\n" as a visual separator
    between turns when loop mode auto-continues. convert_to_openai_messages strips it to
    an empty string, yielding an assistant message with neither content nor tool_calls.
    DeepSeek (and OpenRouter's BYOK relay for it) rejects such a message with a 400
    ("Invalid assistant message: content or tool_calls must be set"). The separator
    carries no information to the model, so it must be dropped from the request.
    """
    messages = [
        {"role": "user", "type": "message", "content": "hi"},
        {"role": "assistant", "type": "message", "content": "Let me do that."},
        {"role": "assistant", "type": "message", "content": "\n\n"},
        {"role": "user", "type": "message", "content": "Proceed."},
    ]
    out = convert_to_openai_messages(
        messages, function_calling=True, vision=False, interpreter=_FakeInterpreter()
    )
    for m in out:
        if m["role"] == "assistant":
            assert m.get("tool_calls") or str(m.get("content", "")).strip(), (
                f"assistant message with neither content nor tool_calls leaked: {m!r}"
            )
    assert len(out) == 3


def test_whitespace_separator_does_not_break_reasoning_propagation():
    """Dropping a whitespace separator must not sever the turn's reasoning_content chain.

    The loop separator can be stored between a reasoning block and the assistant message
    it belongs to. Even though the separator is dropped, the following tool-call message
    must still receive the turn's reasoning_content or DeepSeek returns a 400.
    """
    messages = [
        {"role": "user", "type": "message", "content": "do A"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "Plan. \n\n"},
        {"role": "assistant", "type": "message", "content": "\n\n"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print('A')"},
    ]
    out = convert_to_openai_messages(
        messages, function_calling=True, vision=False, interpreter=_FakeInterpreter()
    )
    function_calls = [m for m in out if "function_call" in m]
    assert len(function_calls) == 1
    assert function_calls[0]["reasoning_content"] == "Plan. \n\n"


def test_user_message_boundary_resets_pending_reasoning():
    """A new user turn must not inherit the previous turn's reasoning_content.

    Reasoning belongs to the turn that produced it. After a user message, the next
    tool-call message carries no reasoning_content (the API-layer fallback pads it with
    "" for turns where the model did not think), so a stale reasoning value must not
    leak across the turn boundary.
    """
    messages = [
        {"role": "user", "type": "message", "content": "hi"},
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "Greet. \n\n"},
        {"role": "assistant", "type": "message", "content": "Hello!"},
        {"role": "user", "type": "message", "content": "compute"},
        {"role": "assistant", "type": "code", "format": "python", "content": "print(1)"},
    ]
    out = convert_to_openai_messages(
        messages, function_calling=True, vision=False, interpreter=_FakeInterpreter()
    )
    function_calls = [m for m in out if "function_call" in m]
    assert len(function_calls) == 1
    assert "reasoning_content" not in function_calls[0]
