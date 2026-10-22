"""Tests for the per-route wire-quirk table and its use in `Llm.run()`.

The table exists so provider behavior is a row, not a substring check: the
same missing-row bug produced the reasoning_content 400 twice, once when the
route was rewritten and once when a bare wire id stopped matching a prefix.
These tests pin the resolution rules and the two quirks that change a request
today (DashScope thinking controls, Zhipu rejecting `stream_options`).
"""

import litellm
import pytest

from interpreter.core.core import OpenInterpreter
from interpreter.core.llm.utils.provider_quirks import DEFAULT, resolve_quirks

MESSAGES = [
    {"role": "system", "type": "message", "content": "s"},
    {"role": "user", "type": "message", "content": "h"},
]


def _interpreter(model, **attrs):
    """An `OpenInterpreter` with a loaded-looking Llm that never hits the network."""
    interpreter = OpenInterpreter()
    llm = interpreter.llm
    llm.model = model
    llm._is_loaded = True
    llm.api_key = "test-key"
    llm.api_base = "https://example.invalid/v1"
    llm.supports_functions = True
    llm.supports_vision = False
    for key, value in attrs.items():
        setattr(llm, key, value)
    return interpreter


def _request_params(interpreter, messages=None):
    """Drive one `run()` to the LiteLLM boundary and return the final params."""
    captured = {}

    def fake_completion(**params):
        captured.update(params)
        return iter(())

    original = litellm.completion
    litellm.completion = fake_completion
    try:
        list(interpreter.llm.run(messages or MESSAGES))
    finally:
        litellm.completion = original
    return captured


def test_dashscope_region_routes_get_dashscope_quirks():
    """Both region aliases must resolve to the same thinking-control contract.

    A row per prefix would let the two regions drift, which is how the original
    DashScope rewrite ended up applying to one spelling and not the other.
    """
    for prefix in ("dashscope-intl", "dashscope-us"):
        quirks = resolve_quirks(f"{prefix}/qwen3-max")
        assert quirks.thinking_param == "enable_thinking"
        assert quirks.thinking_budget_param == "thinking_budget"
        assert quirks.preserve_thinking_param == "preserve_thinking"


def test_glm_under_a_gateway_resolves_to_zhipu_quirks():
    """opencode_go hides the upstream in the wire id; GLM's rules still apply.

    The gateway speaks OpenAI on the wire, so the route prefix says nothing
    about the model. The family in the wire id is the only signal available.
    """
    assert resolve_quirks("opencode_go/glm-5.1", "glm-5.1").stream_options is False


def test_unknown_route_uses_openai_chat_defaults():
    """An unlisted provider must get a working OpenAIShape, not feature loss.

    This is the totality property: a missing row degrades to the conventional
    contract because it is the one LiteLLM already speaks.
    """
    assert resolve_quirks("some-gateway/brand-new-model") is DEFAULT


def test_dashscope_sends_enable_thinking_false_when_reasoning_disabled():
    """Disabling reasoning must reach DashScope, where thinking is on by default.

    `include_reasoning` is OI's portable switch; DashScope spells it
    `enable_thinking` inside `extra_body`, so without this mapping the switch
    silently did nothing for Qwen 3.5+.
    """
    interpreter = _interpreter(
        "dashscope-intl/qwen3-max",
        _dashscope_wire_model="qwen3-max",
        include_reasoning=False,
    )

    params = _request_params(interpreter)

    assert params["extra_body"]["enable_thinking"] is False
    assert "stream_options" in params


def test_dashscope_maps_integer_reasoning_effort_to_thinking_budget():
    """An integer effort is a token budget in DashScope's vocabulary.

    The two APIs agree on the concept but not the spelling; mapping only the
    integer avoids inventing a low/medium/high conversion the server did not
    document.
    """
    interpreter = _interpreter(
        "dashscope-intl/qwen3-max",
        _dashscope_wire_model="qwen3-max",
        reasoning_effort=4096,
    )

    params = _request_params(interpreter)

    assert params["extra_body"]["enable_thinking"] is True
    assert params["extra_body"]["thinking_budget"] == 4096
    assert "reasoning_effort" not in params


def test_dashscope_preserves_thinking_only_when_history_has_reasoning():
    """`preserve_thinking` must follow the history, not a blanket setting.

    Sending it with no prior reasoning asks the model to preserve nothing, and
    some DashScope models reject parameters their mode does not use.
    """
    interpreter = _interpreter(
        "dashscope-intl/qwen3-max",
        _dashscope_wire_model="qwen3-max",
    )
    with_reasoning = MESSAGES + [
        {
            "role": "assistant",
            "type": "message",
            "format": "reasoning",
            "content": "I should run the code.",
        },
        {"role": "assistant", "type": "message", "content": "done"},
    ]

    assert "preserve_thinking" not in _request_params(interpreter).get("extra_body", {})
    assert (
        _request_params(interpreter, with_reasoning)["extra_body"]["preserve_thinking"]
        is True
    )


def test_route_that_rejects_stream_options_does_not_receive_it():
    """Zhipu/GLM rejects `stream_options`; asking for usage must not break chat.

    The cost is a missing token count, which OI already tolerates, versus a
    rejected request, which loses the turn.
    """
    interpreter = _interpreter(
        "opencode_go/glm-5.1",
        _is_opencode_go=True,
        _dashscope_wire_model=None,
    )

    params = _request_params(interpreter)

    assert params["model"] == "glm-5.1"
    assert "stream_options" not in params


def test_default_route_still_sends_stream_options():
    """The OpenAI default keeps usage accounting, so the gate is narrow."""
    interpreter = _interpreter(
        "openai/gpt-4o",
        _is_opencode_go=False,
        _dashscope_wire_model=None,
    )

    assert "stream_options" in _request_params(interpreter)
