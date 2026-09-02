"""Tests for the Llm class.

The Llm class handles model configuration, message formatting, vision detection,
and streaming completions. These tests cover the configuration properties,
model loading, and the message preparation logic.
"""

from unittest import mock

import pytest

from interpreter.core.llm.llm import Llm, fixed_litellm_completions


def _make_llm(interpreter=None):
    """Build an Llm with a mocked interpreter to avoid real API calls."""
    if interpreter is None:
        interpreter = mock.MagicMock()
        interpreter.computer.vision.query = mock.MagicMock()
        interpreter.computer.vision.ocr = mock.MagicMock(return_value="")
        interpreter.shrink_images = True
        interpreter.os = False
        interpreter.verbose = False
        interpreter.debug = False
        interpreter.in_terminal_interface = False
        interpreter.display_message = mock.MagicMock()
        interpreter.max_output = 2800
    return Llm(interpreter)


def test_llm_default_model():
    """Llm defaults to gpt-4o."""
    llm = _make_llm()
    assert llm.model == "gpt-4o"


def test_llm_default_temperature():
    """Llm defaults to temperature 0.0."""
    llm = _make_llm()
    assert llm.temperature == 0.0


def test_llm_supports_vision_defaults_none():
    """supports_vision defaults to None (auto-detect on first run)."""
    llm = _make_llm()
    assert llm.supports_vision is None


def test_llm_supports_functions_defaults_none():
    """supports_functions defaults to None (auto-detect on first run)."""
    llm = _make_llm()
    assert llm.supports_functions is None


def test_llm_vision_renderer_set():
    """vision_renderer is set to computer.vision.query at construction."""
    llm = _make_llm()
    assert llm.vision_renderer == llm.interpreter.computer.vision.query


def test_llm_completions_is_fixed_litellm():
    """completions is set to the fixed_litellm_completions function."""
    llm = _make_llm()
    assert llm.completions is fixed_litellm_completions


def test_llm_model_setter_resets_loaded():
    """Setting model resets _is_loaded so the new model gets loaded."""
    llm = _make_llm()
    llm._is_loaded = True
    llm.model = "gpt-4.1"
    assert llm._is_loaded is False
    assert llm.model == "gpt-4.1"


def test_llm_load_ollama_adds_latest_tag():
    """load() adds :latest to ollama models without a tag."""
    llm = _make_llm()
    llm.model = "ollama/llama3.1"
    with mock.patch("requests.get") as mock_get, mock.patch("requests.post"):
        mock_get.return_value = mock.MagicMock(ok=True, json=lambda: {"models": []})
        llm.load()
    assert llm.model == "ollama/llama3.1:latest"


def test_llm_load_sets_loaded_flag():
    """load() sets _is_loaded to True after loading."""
    llm = _make_llm()
    llm.model = "gpt-4o"
    with mock.patch("interpreter.core.llm.llm.litellm.get_model_info", return_value={"max_input_tokens": 8000}):
        llm.load()
    assert llm._is_loaded is True


def test_llm_load_skips_if_already_loaded():
    """load() returns early if already loaded."""
    llm = _make_llm()
    llm.model = "gpt-4o"
    llm._is_loaded = True
    with mock.patch("interpreter.core.llm.llm.litellm.get_model_info") as mock_info:
        llm.load()
    mock_info.assert_not_called()
    assert llm.model == "gpt-4o"


def test_llm_run_validates_system_message_first():
    """run() asserts the first message has role='system'."""
    llm = _make_llm()
    messages = [{"role": "user", "type": "message", "content": "hi"}]
    with pytest.raises(AssertionError, match="system"):
        list(llm.run(messages))


def test_llm_run_validates_no_system_after_first():
    """run() asserts no message after the first has role='system'."""
    llm = _make_llm()
    messages = [
        {"role": "system", "type": "message", "content": "sys"},
        {"role": "system", "type": "message", "content": "another"},
    ]
    with pytest.raises(AssertionError, match="system"):
        list(llm.run(messages))


def test_llm_run_remaps_claude_35():
    """run() remaps claude-3.5 model names to claude-sonnet-4-6."""
    llm = _make_llm()
    llm.model = "claude-3.5"
    llm._is_loaded = True
    messages = [
        {"role": "system", "type": "message", "content": "sys"},
        {"role": "user", "type": "message", "content": "hi"},
    ]

    def fake_run_text_llm(self, params):
        """Capture the model param and yield nothing."""
        self._captured_model = params.get("model")
        return iter([])

    with mock.patch("interpreter.core.llm.llm.run_text_llm", fake_run_text_llm), \
         mock.patch("interpreter.core.llm.llm.litellm.supports_function_calling", return_value=False), \
         mock.patch("interpreter.core.llm.llm.litellm.supports_vision", return_value=False):
        list(llm.run(messages))
    assert llm.model == "claude-sonnet-4-6"


def test_llm_run_warns_when_max_tokens_exceeds_context(capsys):
    """run() warns and adjusts max_tokens when it exceeds context_window."""
    llm = _make_llm()
    llm._is_loaded = True
    llm.max_tokens = 100000
    llm.context_window = 1000
    messages = [
        {"role": "system", "type": "message", "content": "sys"},
        {"role": "user", "type": "message", "content": "hi"},
    ]

    with mock.patch("interpreter.core.llm.llm.run_text_llm", return_value=iter([])), \
         mock.patch("interpreter.core.llm.llm.litellm.supports_function_calling", return_value=False), \
         mock.patch("interpreter.core.llm.llm.litellm.supports_vision", return_value=False):
        list(llm.run(messages))
    assert llm.max_tokens == int(0.2 * 1000)
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "max_tokens" in captured.out


def test_llm_run_builds_params():
    """run() builds the correct params dict for the completion call."""
    llm = _make_llm()
    llm._is_loaded = True
    llm.api_key = "test-key"
    llm.api_base = "http://localhost:8000/v1"
    llm.max_tokens = 1000
    llm.temperature = 0.5
    messages = [
        {"role": "system", "type": "message", "content": "sys"},
        {"role": "user", "type": "message", "content": "hi"},
    ]

    captured_params = {}

    def fake_run_text_llm(self, params):
        """Capture the params for assertion."""
        captured_params.update(params)
        return iter([])

    with mock.patch("interpreter.core.llm.llm.run_text_llm", fake_run_text_llm), \
         mock.patch("interpreter.core.llm.llm.litellm.supports_function_calling", return_value=False), \
         mock.patch("interpreter.core.llm.llm.litellm.supports_vision", return_value=False):
        list(llm.run(messages))
    assert captured_params["model"] == "gpt-4o"
    assert captured_params["api_key"] == "test-key"
    assert captured_params["api_base"] == "http://localhost:8000/v1"
    assert captured_params["max_tokens"] == 1000
    assert captured_params["temperature"] == 0.5
    assert captured_params["stream"] is True


def test_llm_run_uses_tool_calling_for_function_models():
    """run() routes to run_tool_calling_llm when supports_functions is True."""
    llm = _make_llm()
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    messages = [
        {"role": "system", "type": "message", "content": "sys"},
        {"role": "user", "type": "message", "content": "hi"},
    ]
    with mock.patch(
        "interpreter.core.llm.llm.run_tool_calling_llm", return_value=iter([])
    ) as mock_tool:
        list(llm.run(messages))
    mock_tool.assert_called_once()


def test_llm_run_uses_text_llm_for_non_function_models():
    """run() routes to run_text_llm when supports_functions is False."""
    llm = _make_llm()
    llm._is_loaded = True
    llm.supports_functions = False
    llm.supports_vision = False
    messages = [
        {"role": "system", "type": "message", "content": "sys"},
        {"role": "user", "type": "message", "content": "hi"},
    ]
    with mock.patch(
        "interpreter.core.llm.llm.run_text_llm", return_value=iter([])
    ) as mock_text:
        list(llm.run(messages))
    mock_text.assert_called_once()


def test_llm_max_budget_property():
    """max_budget defaults to None."""
    llm = _make_llm()
    assert llm.max_budget is None


def test_llm_context_window_defaults_none():
    """context_window defaults to None (auto-detect or unset)."""
    llm = _make_llm()
    assert llm.context_window is None


def test_llm_max_tokens_defaults_none():
    """max_tokens defaults to None (auto-detect or unset)."""
    llm = _make_llm()
    assert llm.max_tokens is None


def test_llm_api_base_defaults_none():
    """api_base defaults to None (use provider default)."""
    llm = _make_llm()
    assert llm.api_base is None


def test_llm_api_key_defaults_none():
    """api_key defaults to None (use env var or provider default)."""
    llm = _make_llm()
    assert llm.api_key is None


def test_llm_execution_instructions_set():
    """execution_instructions contains guidance for code execution."""
    llm = _make_llm()
    assert "markdown code block" in llm.execution_instructions


def test_fixed_litellm_completions_adds_stop_for_local():
    """fixed_litellm_completions adds stop tokens for local models."""
    params = {"model": "local/model", "messages": []}
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = iter([])
        list(fixed_litellm_completions(**params))
    call_kwargs = mock_completion.call_args[1]
    assert "<|assistant|>" in call_kwargs.get("stop", [])


def test_fixed_litellm_completions_strips_latest_tag():
    """fixed_litellm_completions strips :latest from the model name."""
    params = {"model": "ollama/llama3.1:latest", "messages": []}
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = iter([])
        list(fixed_litellm_completions(**params))
    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["model"] == "ollama/llama3.1"


def test_fixed_litellm_completions_retries_on_error():
    """fixed_litellm_completions retries up to 4 times on error."""
    params = {"model": "gpt-4o", "messages": []}
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.side_effect = [Exception("fail"), iter([])]
        list(fixed_litellm_completions(**params))
    assert mock_completion.call_count == 2


def test_fixed_litellm_completions_raises_after_max_retries():
    """fixed_litellm_completions raises the first error after exhausting retries."""
    params = {"model": "gpt-4o", "messages": []}
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.side_effect = Exception("persistent failure")
        with pytest.raises(Exception, match="persistent failure"):
            list(fixed_litellm_completions(**params))
    assert mock_completion.call_count == 4
