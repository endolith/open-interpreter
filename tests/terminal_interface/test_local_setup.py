from unittest import mock

import inquirer
import pytest
import subprocess

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
        """Assert the ollama list invocation and return a fake installed-model listing."""
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


def _mock_psutil(monkeypatch, ram_gb, disk_free_gb=50):
    """Pin resource values so download_model's guidance and disk filtering are deterministic."""

    class _Memory:
        total = ram_gb * 1024**3

    class _Disk:
        free = disk_free_gb * 1024**3

    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.psutil.virtual_memory",
        lambda: _Memory(),
    )
    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.psutil.disk_usage",
        lambda _path: _Disk(),
    )


def test_local_setup_llamafile_launches_existing_model(monkeypatch):
    """Selecting an already-downloaded llamafile runs it with server flags."""
    interpreter = _make_interpreter()
    _mock_ram_gb(monkeypatch, 16)
    interpreter.get_oi_dir.return_value = "/tmp/oi"
    monkeypatch.setattr("os.path.exists", lambda _p: True)
    monkeypatch.setattr("os.listdir", lambda _d: ["tiny.llamafile"])

    process = mock.Mock()
    process.stdout = iter(["llama server listening at http://localhost:8080\n"])
    popen = mock.Mock(return_value=process)
    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.subprocess.Popen", popen
    )
    monkeypatch.setattr(
        inquirer,
        "prompt",
        mock.Mock(
            side_effect=[
                {"model": "Llamafile"},
                {"model": "tiny.llamafile"},
            ]
        ),
    )

    result = local_setup(interpreter)

    assert result is interpreter
    assert interpreter.llm.model == "openai/local"
    assert interpreter.llm.api_base == "http://localhost:8080/v1"
    assert interpreter.llm.supports_functions is False
    assert interpreter.llm.api_key == "dummy"
    assert interpreter.llm.temperature == 0
    assert interpreter.llm.context_window == 8000
    # The selected llamafile path is launched with the server flags.
    popen.assert_called_once_with(
        '"/tmp/oi/models/tiny.llamafile" --nobrowser -ngl 9999',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_local_setup_llamafile_downloads_new_model(monkeypatch):
    """Choosing 'Download new model' downloads, chmods, then launches a model."""
    interpreter = _make_interpreter()
    _mock_psutil(monkeypatch, 16, disk_free_gb=50)
    interpreter.get_oi_dir.return_value = "/tmp/oi"
    monkeypatch.setattr("os.path.exists", lambda _p: True)
    monkeypatch.setattr("os.listdir", lambda _d: ["tiny.llamafile"])

    # An existing model is present, so this exercises the "Download new model"
    # option rather than the empty-directory path.
    download = mock.Mock()
    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.wget.download", download
    )
    run = mock.Mock()
    monkeypatch.setattr("interpreter.terminal_interface.local_setup.subprocess.run", run)
    # The freshly downloaded model is launched as a server; mock Popen with a
    # ready-line iterator so the launch is exercised rather than a real shell
    # "not found" that vacuously passes.
    process = mock.Mock()
    process.stdout = iter(["llama server listening at http://localhost:8080\n"])
    popen = mock.Mock(return_value=process)
    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.subprocess.Popen", popen
    )

    # inquirer.prompt: provider, existing-model selection, then model download.
    monkeypatch.setattr(
        inquirer,
        "prompt",
        mock.Mock(
            side_effect=[
                {"model": "Llamafile"},
                {"model": "↓ Download new model"},
                {"model": "Phi-3-mini (2.42GB)"},
            ]
        ),
    )
    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.platform.system",
        lambda: "Linux",
    )

    result = local_setup(interpreter)

    assert result is interpreter
    assert interpreter.llm.model == "openai/local"
    assert interpreter.llm.api_base == "http://localhost:8080/v1"
    model_path = "/tmp/oi/models/Phi-3-mini-4k-instruct.Q4_K_M.llamafile"
    # wget downloads the selected model's URL to the models directory.
    download.assert_called_once_with(
        "https://huggingface.co/Mozilla/Phi-3-mini-4k-instruct-llamafile/resolve/main/Phi-3-mini-4k-instruct.Q4_K_M.llamafile?download=true",
        model_path,
    )
    # The downloaded file is made executable.
    run.assert_called_once_with(["chmod", "+x", model_path], check=True)
    # The downloaded model is launched with the server flags and its stdout is
    # polled until the ready signal.
    popen.assert_called_once_with(
        f'"{model_path}" --nobrowser -ngl 9999',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_local_setup_ollama_download_new_model(monkeypatch):
    """Ollama '↓ Download llama3.1' triggers ollama pull for that model."""
    interpreter = _make_interpreter()
    _mock_ram_gb(monkeypatch, 16)

    run = mock.Mock(
        side_effect=[
            mock.Mock(stdout="NAME\n", returncode=0),  # ollama list
            mock.Mock(returncode=0),  # ollama pull
        ]
    )
    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.subprocess.run", run
    )
    monkeypatch.setattr(
        inquirer,
        "prompt",
        mock.Mock(
            side_effect=[
                {"model": "Ollama"},
                {"name": "↓ Download llama3.1"},
            ]
        ),
    )

    local_setup(interpreter)

    assert interpreter.llm.model == "ollama/llama3.1"
    assert run.call_count == 2
    assert run.call_args_list[0][0] == (["ollama", "list"],)
    assert run.call_args_list[1][0] == (["ollama", "pull", "llama3.1"],)
    assert run.call_args_list[1].kwargs["check"] is True


def test_local_setup_jan_custom_model_id(monkeypatch):
    """Choosing the custom-model-id entry prompts for the raw id."""
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
                {"jan_model_name": ">> Type Custom Model ID"},
            ]
        ),
    )
    monkeypatch.setattr(
        "interpreter.terminal_interface.local_setup.requests.get",
        lambda *args, **kwargs: _Response(),
    )
    with mock.patch("builtins.input", return_value="my-custom-model"):
        local_setup(interpreter)

    assert interpreter.llm.model == "my-custom-model"
