from unittest import mock

import inquirer
import pytest

from interpreter.terminal_interface.local_setup import local_setup


def _make_interpreter(*, auto_run=True):
    """Minimal interpreter stub for local_setup provider branches."""
    interpreter = mock.MagicMock()
    interpreter.auto_run = auto_run
    interpreter.llm = mock.MagicMock()
    interpreter.llm.max_tokens = 100
    interpreter.llm.context_window = 100
    return interpreter


def _mock_ram_gb(monkeypatch, gb):
    """Pin reported RAM so local_setup picks predictable context limits."""

    class _Memory:
        total = gb * 1024**3

    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.psutil.virtual_memory",
        lambda: _Memory(),
    )


def test_local_setup_lm_studio_sets_openai_compatible_api(monkeypatch):
    """LM Studio wires a local OpenAI-compatible API and disables function calling."""
    interpreter = _make_interpreter()
    _mock_ram_gb(monkeypatch, 16)
    monkeypatch.setattr(
        inquirer, "prompt", lambda questions: {"model": "LM Studio"}
    )

    result = local_setup(interpreter)

    assert result is interpreter
    assert interpreter.llm.api_base == "http://localhost:1234/v1"
    assert interpreter.llm.api_key == "dummy"
    assert interpreter.llm.supports_functions is False
    assert interpreter.llm.context_window == 8000
    assert interpreter.llm.max_tokens == 1200


def test_local_setup_jan_sets_model_from_api(monkeypatch):
    """Jan fetches /models, prompts for a model id, then applies LLM settings."""
    interpreter = _make_interpreter()
    _mock_ram_gb(monkeypatch, 16)

    class _Response:
        def json(self):
            return {"data": [{"id": "jan-model-1"}]}

    monkeypatch.setattr(
        inquirer,
        "prompt",
        mock.Mock(
            side_effect=[
                {"model": "Jan"},
                {"jan_model_name": "jan-model-1"},
            ]
        ),
    )
    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.requests.get",
        lambda *args, **kwargs: _Response(),
    )

    result = local_setup(interpreter)

    assert result is interpreter
    assert interpreter.llm.api_base == "http://localhost:1337/v1"
    assert interpreter.llm.model == "jan-model-1"
    assert interpreter.llm.api_key == "dummy"


def test_local_setup_ollama_selects_model_and_pings(monkeypatch):
    """Ollama lists local models, applies ollama/ prefix, and sends a warm-up ping."""
    interpreter = _make_interpreter()
    _mock_ram_gb(monkeypatch, 16)

    def _ollama_list(*args, **kwargs):
        assert args[0] == ["ollama", "list"]
        return mock.Mock(
            stdout="NAME\nllama3.1:latest\n",
            returncode=0,
        )

    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.subprocess.run",
        _ollama_list,
    )
    monkeypatch.setattr(
        inquirer,
        "prompt",
        mock.Mock(
            side_effect=[
                {"model": "Ollama"},
                {"name": "llama3.1"},
            ]
        ),
    )

    result = local_setup(interpreter)

    assert result is interpreter
    assert interpreter.llm.model == "ollama/llama3.1"
    interpreter.computer.ai.chat.assert_called_once_with("ping")
    assert interpreter.llm.context_window == 8000


def test_local_setup_ollama_missing_exits_when_not_installed(monkeypatch):
    """Missing ollama binary prints install guidance and exits."""
    interpreter = _make_interpreter()
    _mock_ram_gb(monkeypatch, 16)
    monkeypatch.setattr(
        inquirer, "prompt", lambda questions: {"model": "Ollama"}
    )
    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.subprocess.run",
        mock.Mock(side_effect=FileNotFoundError("ollama")),
    )
    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.time.sleep",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(SystemExit) as exc_info:
        local_setup(interpreter)

    assert exc_info.value.code == 1


def test_local_setup_low_ram_uses_smaller_context_window(monkeypatch):
    """Machines with <=9GB RAM get the smaller default context window."""
    interpreter = _make_interpreter()
    _mock_ram_gb(monkeypatch, 8)
    monkeypatch.setattr(
        inquirer, "prompt", lambda questions: {"model": "LM Studio"}
    )

    local_setup(interpreter)

    assert interpreter.llm.context_window == 3000
    assert interpreter.llm.max_tokens == 1000
