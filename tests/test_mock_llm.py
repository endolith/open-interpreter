import pytest

from interpreter import OpenInterpreter
from tests.support.mock_openai_server import MockOpenAIServer


pytestmark = pytest.mark.mock_llm


@pytest.fixture
def mock_llm_server():
    """Start a scenario-based local OpenAI-compatible server for the duration of a test."""
    server = MockOpenAIServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _mock_interpreter(server: MockOpenAIServer, *, auto_run: bool = False) -> OpenInterpreter:
    """OpenInterpreter wired to the mock server instead of a real LLM provider."""
    interpreter = OpenInterpreter(disable_telemetry=True)
    interpreter.auto_run = auto_run
    interpreter.llm.model = "openai/gpt-4o-mini"
    interpreter.llm.api_base = server.api_base
    interpreter.llm.api_key = "mock-key"
    interpreter.llm.supports_functions = False
    interpreter.llm._is_loaded = False
    return interpreter


@pytest.mark.timeout(60)
def test_chat_uses_mock_openai_api(mock_llm_server):
    """chat() completes against a local OpenAI-compatible server without a real API key."""
    interpreter = _mock_interpreter(mock_llm_server)

    messages = interpreter.chat("Say hello.", display=False, stream=False, blocking=True)

    assert messages[-1]["content"] == "Hello, World!"


@pytest.mark.timeout(60)
def test_mock_llm_hello_world(mock_llm_server):
    """Scenario mock returns exactly Hello, World! for the integration-style prompt."""
    interpreter = _mock_interpreter(mock_llm_server)
    prompt = (
        "Please reply with just the words Hello, World! and nothing else. "
        "Do not run code. No confirmation just the text."
    )

    messages = interpreter.chat(prompt, display=False, stream=False, blocking=True)

    assert messages == [
        {"role": "assistant", "type": "message", "content": "Hello, World!"}
    ]


@pytest.mark.timeout(60)
def test_mock_llm_write_to_file(mock_llm_server, monkeypatch, tmp_path):
    """Scenario mock returns Python that writes a file; chat() auto-runs it without an API key."""
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_interpreter(mock_llm_server, auto_run=True)

    interpreter.chat(
        "Write the word 'Washington' to a .txt file called file.txt. "
        "Instantly run the code! Save the file!",
        display=False,
        stream=False,
        blocking=True,
    )

    assert (tmp_path / "file.txt").read_text() == "Washington"

    interpreter.messages = []
    messages = interpreter.chat(
        "Read file.txt in the current directory and tell me what's in it.",
        display=False,
        stream=False,
        blocking=True,
    )

    assert "Washington" in messages[-1]["content"]
