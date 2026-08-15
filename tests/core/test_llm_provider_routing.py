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
