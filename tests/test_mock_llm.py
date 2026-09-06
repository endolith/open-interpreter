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


def _mock_tool_interpreter(server: MockOpenAIServer) -> OpenInterpreter:
    """OpenInterpreter in function-calling mode against the mock server.

    supports_functions routes llm.run() through run_tool_calling_llm, so the
    full HTTP request → streaming tool_calls deltas → parse → execute path is
    exercised. Loading is skipped for determinism (no model-info lookup).
    """
    interpreter = OpenInterpreter(disable_telemetry=True)
    interpreter.auto_run = True
    interpreter.llm.model = "openai/gpt-4o-mini"
    interpreter.llm.api_base = server.api_base
    interpreter.llm.api_key = "mock-key"
    interpreter.llm.supports_functions = True
    interpreter.llm.supports_vision = False
    interpreter.llm._is_loaded = True
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


_ERRAND_PROMPT = (
    "Please run this errand: write step one, then step two, then report back. "
    "Start now."
)


def _assert_errand_complete(interpreter, tmp_path, messages):
    """The errand ran python then shell, in that order, and ended by talking."""
    assert (tmp_path / "step1.txt").read_text() == "one"
    assert (tmp_path / "step2.txt").read_text().strip() == "two"
    assert "Errand complete." in messages[-1]["content"]
    formats = [m.get("format") for m in messages if m.get("type") == "code"]
    assert formats == ["python", "shell"]


@pytest.mark.timeout(120)
def test_mock_llm_tool_call_errand(mock_llm_server, monkeypatch, tmp_path):
    """One tool-calling convo alternates talking and executing, in two languages.

    The mock server emits real OpenAI streaming tool_calls deltas (split across
    chunks); the run executes python, then shell, then ends by talking — all
    through HTTP with no API key.
    """
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    messages = interpreter.chat(
        _ERRAND_PROMPT, display=False, stream=False, blocking=True
    )

    _assert_errand_complete(interpreter, tmp_path, messages)


@pytest.mark.timeout(120)
def test_mock_llm_text_errand(mock_llm_server, monkeypatch, tmp_path):
    """The same errand in code-block mode: fences, two languages, then talking."""
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_interpreter(mock_llm_server, auto_run=True)

    messages = interpreter.chat(
        _ERRAND_PROMPT, display=False, stream=False, blocking=True
    )

    _assert_errand_complete(interpreter, tmp_path, messages)


@pytest.mark.timeout(120)
def test_mock_llm_tool_errand_after_prior_chat(mock_llm_server, monkeypatch, tmp_path):
    """An errand started after an unrelated chat still begins at the python step.

    Turn counting is scoped to messages after the errand prompt; the assistant
    turn from the earlier hello chat must not shift the errand to turn 1.
    """
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    interpreter.chat("Say hello.", display=False, stream=False, blocking=True)
    messages = interpreter.chat(
        _ERRAND_PROMPT, display=False, stream=False, blocking=True
    )

    _assert_errand_complete(interpreter, tmp_path, messages)


@pytest.mark.timeout(60)
def test_mock_llm_auth_text_unaffected(mock_llm_server, monkeypatch):
    """INTERPRETER_REQUIRE_AUTHENTICATION does not break tool-less runs.

    The auth judge-layer guard only applies when a function call was detected;
    plain talking must pass through unchanged with enforcement enabled.
    """
    monkeypatch.setenv("INTERPRETER_REQUIRE_AUTHENTICATION", "true")
    interpreter = _mock_interpreter(mock_llm_server)

    messages = interpreter.chat("Say hello.", display=False, stream=False, blocking=True)

    assert messages[-1]["content"] == "Hello, World!"
