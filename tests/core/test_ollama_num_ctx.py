"""Regression tests for the Ollama num_ctx default (PR #271)."""

from interpreter.core.core import OpenInterpreter


def _request_params(model, context_window):
    """Drive one Llm.run() with stubbed completions; return its request params."""
    captured = {}

    def fake_completions(**params):
        captured.update(params)
        return iter(())

    interpreter = OpenInterpreter()
    interpreter.llm.model = model
    interpreter.llm.context_window = context_window
    interpreter.llm.supports_functions = False
    # Skip load(): it probes the local Ollama server over HTTP and exits
    # when Ollama is not installed. The num_ctx logic lives in run().
    interpreter.llm._is_loaded = True
    interpreter.llm.completions = fake_completions
    list(
        interpreter.llm.run(
            [
                {"role": "system", "type": "message", "content": "s"},
                {"role": "user", "type": "message", "content": "h"},
            ]
        )
    )
    return captured


def test_ollama_model_sends_num_ctx():
    """An ollama/ model sends num_ctx matching context_window.

    Ollama sizes its KV cache from num_ctx; without it the server runs at
    its own default context while OI trims to context_window, silently
    overflowing the allocation when context_window is larger.
    """
    params = _request_params("ollama/llama3", 8192)
    assert params["num_ctx"] == 8192


def test_ollama_chat_model_sends_num_ctx():
    """An ollama_chat/ model sends num_ctx matching context_window.

    Both LiteLLM Ollama flavors read the same option, so both prefixes
    must be covered.
    """
    params = _request_params("ollama_chat/llama3", 8192)
    assert params["num_ctx"] == 8192


def test_non_ollama_model_sends_no_num_ctx():
    """A non-Ollama model never receives num_ctx.

    num_ctx is Ollama-specific; other providers must not see the key.
    """
    params = _request_params("gpt-4o-mini", 128000)
    assert "num_ctx" not in params


def test_unset_context_window_sends_no_num_ctx():
    """No num_ctx is sent when context_window is unset.

    There is nothing to agree on, so the key must stay absent rather than
    sending a null that a strict server could reject.
    """
    params = _request_params("ollama/llama3", None)
    assert "num_ctx" not in params
