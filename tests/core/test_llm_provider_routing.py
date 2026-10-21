import os

import pytest

import litellm
import requests
import interpreter.core.llm.llm as llm_mod
from interpreter.core.core import OpenInterpreter


@pytest.fixture
def interpreter():
    return OpenInterpreter()


def test_deepseek_load_sets_api_defaults(interpreter, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    interpreter.llm.model = "deepseek/deepseek-v4-flash"
    interpreter.llm.api_key = None
    interpreter.llm.api_base = None
    interpreter.llm._is_loaded = False

    interpreter.llm.load()

    assert interpreter.llm.model == "deepseek/deepseek-v4-flash"
    assert interpreter.llm.api_key == "sk-test-deepseek"
    assert interpreter.llm.api_base == "https://api.deepseek.com"


def test_deepseek_load_uses_default_base_without_env(interpreter, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    interpreter.llm.model = "deepseek/deepseek-v4-pro"
    interpreter.llm.api_key = None
    interpreter.llm.api_base = None
    interpreter.llm._is_loaded = False

    interpreter.llm.load()

    assert interpreter.llm.api_base == "https://api.deepseek.com"
    assert interpreter.llm.api_key is None


def test_dashscope_intl_load_keeps_truthful_model_and_parks_wire_name(
    interpreter, monkeypatch
):
    """The region alias stays the route name; only the wire name is bare.

    Rewriting self.model to openai/<model> destroyed the route identity every
    gate reads, and it bought no metadata: LiteLLM's registry knows the model as
    dashscope/qwen3-max (tools + context window), while openai/qwen3-max is a
    total miss -- so the rewrite silently disabled tool calling for this route.
    """
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-dashscope")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    interpreter.llm.model = "dashscope-intl/qwen3-max"
    interpreter.llm.api_key = None
    interpreter.llm.api_base = None
    interpreter.llm._is_loaded = False

    interpreter.llm.load()

    assert interpreter.llm.model == "dashscope-intl/qwen3-max"
    assert interpreter.llm.wire_model_name() == "qwen3-max"
    assert interpreter.llm.wire_provider() == "openai"
    assert interpreter.llm.api_key == "sk-test-dashscope"
    assert interpreter.llm.api_base == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def test_dashscope_us_load_keeps_truthful_model_and_parks_wire_name(
    interpreter, monkeypatch
):
    """The US alias behaves like the intl one, and keeps qwen3.5 vision enabled."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-dashscope")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    interpreter.llm.model = "dashscope-us/qwen3.5-plus"
    interpreter.llm.api_key = None
    interpreter.llm.api_base = None
    interpreter.llm.supports_vision = None
    interpreter.llm._is_loaded = False

    interpreter.llm.load()

    assert interpreter.llm.model == "dashscope-us/qwen3.5-plus"
    assert interpreter.llm.wire_model_name() == "qwen3.5-plus"
    assert interpreter.llm.api_key == "sk-test-dashscope"
    assert interpreter.llm.api_base == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    assert interpreter.llm.supports_vision is True



class _FakeOpenRouterResponse:
    def __init__(self, modalities_by_id):
        self.modalities_by_id = modalities_by_id

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "data": [
                {"id": mid, "architecture": {"input_modalities": mods}} for mid, mods in self.modalities_by_id.items()
            ]
        }


@pytest.fixture
def stub_openrouter_registry(monkeypatch):
    """Simulate a stale LiteLLM registry and a stubbed provider call."""
    monkeypatch.setattr(litellm, "supports_vision", lambda model: False)
    monkeypatch.setattr(
        llm_mod,
        "run_text_llm",
        lambda self, params: iter([("message", {"role": "assistant", "type": "message", "content": "stubbed"})]),
    )
    return monkeypatch


def _run_one_turn(interpreter):
    messages = [
        {"role": "system", "type": "message", "content": "You are helpful."},
        {"role": "user", "type": "message", "content": "hi"},
    ]
    next(interpreter.llm.run(messages))


def test_openrouter_qwen37_vision_detected_when_registry_stale(interpreter, stub_openrouter_registry):
    stub_openrouter_registry.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeOpenRouterResponse({"qwen/qwen3.7-plus": ["text", "image"]}),
    )

    interpreter.llm.model = "openrouter/qwen/qwen3.7-plus"
    interpreter.llm.supports_vision = None
    interpreter.llm._is_loaded = False

    _run_one_turn(interpreter)

    assert interpreter.llm.supports_vision is True


def test_openrouter_qwen37_text_only_not_vision(interpreter, stub_openrouter_registry):
    stub_openrouter_registry.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeOpenRouterResponse({"qwen/qwen3.7-max": ["text"]}),
    )

    interpreter.llm.model = "openrouter/qwen/qwen3.7-max"
    interpreter.llm.supports_vision = None
    interpreter.llm._is_loaded = False

    _run_one_turn(interpreter)

    assert interpreter.llm.supports_vision is False


def test_openrouter_vision_helper_skips_non_openrouter_models(interpreter, stub_openrouter_registry):
    interpreter.llm.model = "deepseek/deepseek-v4-flash"
    interpreter.llm.supports_vision = None
    interpreter.llm._is_loaded = False

    _run_one_turn(interpreter)

    assert interpreter.llm.supports_vision is False


_DEEPSEEK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute",
            "description": "run code",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@pytest.fixture
def capture_deepseek_params(monkeypatch):
    """Capture the request params passed to litellm.completion for the deepseek pass.

    Passback is conditional on the request carrying `tools` (DeepSeek's thinking
    mode guide), so the default models the real failing shape; use tools=None for
    a request without them.
    """

    def _capture(
        messages, model="openrouter/deepseek/deepseek-v4-flash", tools=_DEEPSEEK_TOOLS
    ):
        captured = {}

        def fake_completion(**params):
            captured["params"] = params
            return iter([])

        kwargs = {"model": model, "messages": messages}
        if tools is not None:
            kwargs["tools"] = tools

        monkeypatch.setattr(litellm, "completion", fake_completion)
        list(llm_mod.fixed_litellm_completions(**kwargs))
        return captured["params"]["messages"]

    return _capture


def test_deepseek_reasoning_pads_synthetic_assistant_with_real_reasoning(
    capture_deepseek_params,
):
    """The API-layer fallback propagates the turn's real reasoning, never "" on a thought turn.

    process_messages can inject synthetic assistant tool_calls messages that lack
    reasoning_content. Patching them with "" (the old behavior) is rejected with a 400
    when the model actually reasoned that turn, so the fallback must copy the current
    turn's reasoning instead.
    """
    out = capture_deepseek_params(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "compute 2+2"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Let me compute. \n\n",
                "function_call": {"name": "execute", "arguments": "{}"},
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "execute", "arguments": "{}"},
                    }
                ],
            },
        ]
    )
    assert out[2]["reasoning_content"] == "Let me compute. \n\n"
    assert out[3]["reasoning_content"] == "Let me compute. \n\n"


def test_deepseek_reasoning_pads_empty_when_model_did_not_think(
    capture_deepseek_params,
):
    """Turns where the model produced no reasoning still get reasoning_content = "".

    DeepSeek requires the field to exist on every assistant message in thinking mode,
    but accepts an empty string for turns where the model did not think. The fallback
    must keep that "" default (it is only wrong on turns that actually reasoned).
    """
    out = capture_deepseek_params(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
    )
    assert out[2]["reasoning_content"] == ""


def test_deepseek_reasoning_placeholder_for_empty_tool_call_turn(
    capture_deepseek_params,
):
    """A tool-call message with no recorded reasoning gets a non-empty placeholder.

    DeepSeek rejects BOTH an empty string and a missing reasoning_content on an
    assistant tool_calls message when the request carries tools and thinking mode is
    enabled — only a non-empty value is accepted. Turns where the model produced no
    thinking (e.g. a trivial "ok" -> execute code) have no recoverable reasoning, so
    the fallback synthesizes a neutral placeholder instead of emitting "".
    """
    out = capture_deepseek_params(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "ok"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "execute", "arguments": "{}"},
                    }
                ],
            },
        ]
    )
    assert out[2]["reasoning_content"] == "."


def test_deepseek_reasoning_placeholder_replaces_prefilled_empty_string(
    capture_deepseek_params,
):
    """A tool-call message carrying "" from the converter is normalized too.

    convert_to_openai_messages can attach reasoning_content="" to a tool-call message
    directly (not just leave it missing). Both shapes must be normalized to the
    placeholder, since DeepSeek rejects "" on tool_calls just the same.
    """
    out = capture_deepseek_params(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "ok"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "execute", "arguments": "{}"},
                    }
                ],
            },
        ]
    )
    assert out[2]["reasoning_content"] == "."


def test_deepseek_reasoning_does_not_leak_across_user_turn(
    capture_deepseek_params,
):
    """The fallback resets its last-seen reasoning at a user boundary.

    A tool-call message in a fresh user turn that happens to lack reasoning_content must
    not inherit the reasoning from a previous turn — it is a no-thought turn and gets the
    neutral placeholder instead, since DeepSeek rejects "" on tool_calls messages.
    """
    out = capture_deepseek_params(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "Greet. \n\n"},
            {"role": "user", "content": "compute 1+1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "execute", "arguments": "{}"},
                    }
                ],
            },
        ]
    )
    assert out[2]["reasoning_content"] == "Greet. \n\n"
    assert out[4]["reasoning_content"] == "."


def test_deepseek_reasoning_padded_for_opencode_go_route(capture_deepseek_params):
    """DeepSeek behind the OpenCode Go gateway also needs reasoning_content.

    The gateway relays DeepSeek with the same thinking mode, and a request that
    carries tools plus an assistant tool_calls message without the field comes
    back as a bare "400 {'model': 'deepseek-v4.1-flash'}" with no explanation.

    The opencode_go/ prefix used to be rewritten to openai/deepseek-* before
    this point, which is why the guard originally needed an "openai/"
    alternative. The prefix is now kept in self.model, so the route matches
    directly, and the rewrite is applied only to the params sent to LiteLLM.
    """
    out = capture_deepseek_params(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "try cmd"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "execute", "arguments": "{}"},
                    }
                ],
            },
        ],
        model="opencode_go/deepseek-v4.1-flash",
    )
    assert out[2]["reasoning_content"] == "."


def test_deepseek_reasoning_padding_skipped_for_other_openai_models(
    capture_deepseek_params,
):
    """Non-DeepSeek models on the openai/ route must not get reasoning_content.

    The opencode_go/ guard matches its own prefix, so genuine OpenAI models (and
    arbitrary api_base overrides) must stay untouched — some reject unknown
    message keys. Matching is by substring within the prefix, as it already is
    for openrouter/, so a Go model with "deepseek" in its name is treated as
    DeepSeek-family.
    """
    for model in ("openai/gpt-4o", "openai/o3-mini", "openai/glm-4.6"):
        out = capture_deepseek_params(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            model=model,
        )
        assert "reasoning_content" not in out[2], model


def test_deepseek_reasoning_padding_skipped_for_non_deepseek_models(
    capture_deepseek_params,
):
    """Assistant messages are left untouched for models that do not require the field.

    Only DeepSeek (and DeepSeek-behind-OpenRouter) demand reasoning_content on every
    assistant message. Other models must not get the field injected, since their APIs
    may reject unknown keys.
    """
    out = capture_deepseek_params(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        model="openrouter/anthropic/claude-sonnet-4-6",
    )
    assert "reasoning_content" not in out[2]


def test_deepseek_reasoning_legacy_placeholder_not_propagated(
    capture_deepseek_params,
):
    """A stored legacy placeholder is treated as "no reasoning" and never forwarded.

    The old placeholder text ("Executing the requested command.") was injected into
    outgoing requests as the model's own prior reasoning. DeepSeek's interleaved
    thinking mode echoed it back and it got persisted into conversations as real
    reasoning, teaching the model to narrate an action without executing it. When that
    contamination is already in stored history, the fallback must (a) recognize the
    legacy string as a placeholder, not real thinking, and (b) refuse to propagate it
    onto later assistant messages as if the model had thought it.
    """
    out = capture_deepseek_params(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "ok"},
            {
                "role": "assistant",
                "content": "Executing.",
                "reasoning_content": "Executing the requested command.\n\n",
            },
            # A later tool-call in the same turn must NOT inherit the placeholder as
            # reasoning — it is a no-thought turn, so it gets the inert placeholder.
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "execute", "arguments": "{}"},
                    }
                ],
            },
        ]
    )
    assert out[2]["reasoning_content"] == ""
    assert out[3]["reasoning_content"] == "."


def test_deepseek_reasoning_placeholder_is_semantically_inert(
    capture_deepseek_params,
):
    """The no-thinking placeholder must not read as an action already performed.

    The previous placeholder "Executing the requested command." was sent to the API as
    the model's own prior reasoning, and DeepSeek echoed it back as real reasoning,
    which taught the model that announcing an action without executing it is fine. The
    replacement must be a lone period: non-empty (so the API accepts it on tool_calls)
    but with no content for the model to imitate or echo.
    """
    out = capture_deepseek_params(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "ok"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "execute", "arguments": "{}"},
                    }
                ],
            },
        ]
    )
    assert out[2]["reasoning_content"] == "."


def test_opencode_go_run_parks_wire_name_for_transport(interpreter):
    """run() must leave the truthful route name in params["model"].

    The bare wire id is parked under a private key for
    fixed_litellm_completions to apply at send time. Writing it into
    params["model"] here made every downstream prefix gate read a model string
    that no longer carried the route, which is how the bare
    "deepseek-v4.1-flash" lost its reasoning_content fill and 400d.
    """
    llm = interpreter.llm
    llm.model = "opencode_go/deepseek-v4.1-flash"
    llm._is_opencode_go = True
    llm._is_loaded = True
    llm.api_key = "x"
    llm.api_base = "https://example.invalid/v1"
    llm.supports_functions = True
    llm.supports_vision = False
    captured = {}

    def fake_completions(**params):
        captured.update(params)
        return iter(())

    llm.completions = fake_completions
    list(
        llm.run(
            [
                {"role": "system", "content": "You are helpful.", "type": "message"},
                {"role": "user", "content": "hello", "type": "message"},
            ]
        )
    )
    assert captured["model"] == "opencode_go/deepseek-v4.1-flash"
    assert captured["_oi_wire_model"] == "deepseek-v4.1-flash"
    assert captured["custom_llm_provider"] == "openai"


def test_dashscope_request_carries_bare_wire_name_and_enables_tools(
    interpreter, monkeypatch
):
    """DashScope keeps tool calling now that metadata is looked up as dashscope/*.

    Drives run() to the LiteLLM boundary. The request must carry the bare model
    id plus custom_llm_provider=openai, and a tools schema must be present:
    supports_functions was previously resolved against openai/<model>, a miss
    that silently sent these models down the text path with no tool schema.
    """
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-dashscope")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    interpreter.llm.model = "dashscope-intl/qwen3-max"
    interpreter.llm.api_key = None
    interpreter.llm.api_base = None
    interpreter.llm.supports_functions = None
    interpreter.llm._is_loaded = False
    interpreter.llm.load()
    interpreter.llm.api_base = "https://example.invalid/v1"
    captured = {}

    def fake_completion(**params):
        captured.update(params)
        return iter(())

    original = litellm.completion
    litellm.completion = fake_completion
    try:
        list(
            interpreter.llm.run(
                [
                    {"role": "system", "type": "message", "content": "s"},
                    {"role": "user", "type": "message", "content": "h"},
                ]
            )
        )
    finally:
        litellm.completion = original

    assert captured["model"] == "qwen3-max"
    assert captured["custom_llm_provider"] == "openai"
    assert interpreter.llm.supports_functions is True
    assert "tools" in captured


def test_wire_model_name_is_identity_for_unrouted_models(interpreter):
    """A model LiteLLM can route by itself must be sent unchanged."""
    interpreter.llm.model = "openai/gpt-4o"
    interpreter.llm._dashscope_wire_model = None
    interpreter.llm._is_opencode_go = False

    assert interpreter.llm.wire_model_name() == "openai/gpt-4o"
    assert interpreter.llm.wire_provider() is None


def test_wire_model_name_strips_opencode_go_prefix(interpreter):
    """The Go gateway takes the bare id plus an explicit provider."""
    interpreter.llm.model = "opencode_go/deepseek-v4.1-flash"
    interpreter.llm._is_opencode_go = True
    interpreter.llm._dashscope_wire_model = None

    assert interpreter.llm.wire_model_name() == "deepseek-v4.1-flash"
    assert interpreter.llm.wire_provider() == "openai"
