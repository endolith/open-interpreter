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


def test_dashscope_intl_load_rewrites_to_openai_compatible(interpreter, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-dashscope")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    interpreter.llm.model = "dashscope-intl/qwen3-max"
    interpreter.llm.api_key = None
    interpreter.llm.api_base = None
    interpreter.llm._is_loaded = False

    interpreter.llm.load()

    assert interpreter.llm.model == "openai/qwen3-max"
    assert interpreter.llm.api_key == "sk-test-dashscope"
    assert (
        interpreter.llm.api_base
        == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )


def test_dashscope_us_load_rewrites_to_openai_compatible(interpreter, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-dashscope")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    interpreter.llm.model = "dashscope-us/qwen3.5-plus"
    interpreter.llm.api_key = None
    interpreter.llm.api_base = None
    interpreter.llm.supports_vision = None
    interpreter.llm._is_loaded = False

    interpreter.llm.load()

    assert interpreter.llm.model == "openai/qwen3.5-plus"
    assert interpreter.llm.api_key == "sk-test-dashscope"
    assert (
        interpreter.llm.api_base
        == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    )
    assert interpreter.llm.supports_vision is True


class _FakeOpenRouterResponse:
    def __init__(self, modalities_by_id):
        self.modalities_by_id = modalities_by_id

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "data": [
                {"id": mid, "architecture": {"input_modalities": mods}}
                for mid, mods in self.modalities_by_id.items()
            ]
        }


@pytest.fixture
def stub_openrouter_registry(monkeypatch):
    """Simulate a stale LiteLLM registry and a stubbed provider call."""
    monkeypatch.setattr(litellm, "supports_vision", lambda model: False)
    monkeypatch.setattr(
        llm_mod,
        "run_text_llm",
        lambda self, params: iter(
            [("message", {"role": "assistant", "type": "message", "content": "stubbed"})]
        ),
    )
    return monkeypatch


def _run_one_turn(interpreter):
    messages = [
        {"role": "system", "type": "message", "content": "You are helpful."},
        {"role": "user", "type": "message", "content": "hi"},
    ]
    next(interpreter.llm.run(messages))


def test_openrouter_qwen37_vision_detected_when_registry_stale(
    interpreter, stub_openrouter_registry
):
    stub_openrouter_registry.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeOpenRouterResponse(
            {"qwen/qwen3.7-plus": ["text", "image"]}
        ),
    )

    interpreter.llm.model = "openrouter/qwen/qwen3.7-plus"
    interpreter.llm.supports_vision = None
    interpreter.llm._is_loaded = False

    _run_one_turn(interpreter)

    assert interpreter.llm.supports_vision is True


def test_openrouter_qwen37_text_only_not_vision(
    interpreter, stub_openrouter_registry
):
    stub_openrouter_registry.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeOpenRouterResponse(
            {"qwen/qwen3.7-max": ["text"]}
        ),
    )

    interpreter.llm.model = "openrouter/qwen/qwen3.7-max"
    interpreter.llm.supports_vision = None
    interpreter.llm._is_loaded = False

    _run_one_turn(interpreter)

    assert interpreter.llm.supports_vision is False


def test_openrouter_vision_helper_skips_non_openrouter_models(
    interpreter, stub_openrouter_registry
):
    interpreter.llm.model = "deepseek/deepseek-v4-flash"
    interpreter.llm.supports_vision = None
    interpreter.llm._is_loaded = False

    _run_one_turn(interpreter)

    assert interpreter.llm.supports_vision is False


@pytest.fixture
def capture_deepseek_params(monkeypatch):
    """Capture the request params passed to litellm.completion for the deepseek pass."""

    def _capture(messages, model="openrouter/deepseek/deepseek-v4-flash"):
        captured = {}

        def fake_completion(**params):
            captured["params"] = params
            return iter([])

        monkeypatch.setattr(litellm, "completion", fake_completion)
        list(llm_mod.fixed_litellm_completions(model=model, messages=messages))
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
