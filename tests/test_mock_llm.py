import pytest

from interpreter import OpenInterpreter
from tests.support.mock_openai_server import MockOpenAIServer

pytestmark = pytest.mark.mock_llm


@pytest.fixture
def mock_llm_server():
    """Start a local OpenAI-compatible server for the duration of a test."""
    server = MockOpenAIServer(reply_text="Hello, World!")
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.timeout(60)
def test_chat_uses_mock_openai_api(mock_llm_server):
    """chat() completes against a local OpenAI-compatible server without a real API key."""
    interpreter = OpenInterpreter(disable_telemetry=True)
    interpreter.llm.model = "openai/gpt-4o-mini"
    interpreter.llm.api_base = mock_llm_server.api_base
    interpreter.llm.api_key = "mock-key"
    interpreter.llm.supports_functions = False
    interpreter.llm._is_loaded = False

    messages = interpreter.chat("Say hello.", display=False, stream=False, blocking=True)

    assert messages[-1]["content"] == "Hello, World!"
