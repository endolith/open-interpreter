import pytest

from interpreter import OpenInterpreter
from tests.support.mock_openai_server import MockOpenAIServer

pytestmark = pytest.mark.mock_llm


@pytest.fixture
def mock_llm_server():
    server = MockOpenAIServer(reply_text="Hello, World!")
    server.start()
    try:
        yield server
    finally:
        server.stop()


def test_chat_uses_mock_openai_api(mock_llm_server):
    interpreter = OpenInterpreter()
    interpreter.llm.model = "openai/gpt-4o-mini"
    interpreter.llm.api_base = mock_llm_server.api_base
    interpreter.llm.api_key = "mock-key"
    interpreter.llm.supports_functions = False
    interpreter.llm._is_loaded = False

    messages = interpreter.chat(
        "Please reply with just the words Hello, World! and nothing else. "
        "Do not run code."
    )

    assert messages[-1]["content"] == "Hello, World!"
