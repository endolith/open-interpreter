import os

import pytest

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


def test_dashscope_us_load_rewrites_to_openai_compatible(interpreter, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-dashscope")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    interpreter.llm.model = "dashscope-us/qwen3.5-plus"
    interpreter.llm.api_key = None
    interpreter.llm.api_base = None
    interpreter.llm._is_loaded = False

    interpreter.llm.load()

    assert interpreter.llm.model == "openai/qwen3.5-plus"
    assert interpreter.llm.api_key == "sk-test-dashscope"
    assert (
        interpreter.llm.api_base
        == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    )
