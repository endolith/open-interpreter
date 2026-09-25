import pytest

from interpreter import OpenInterpreter
from tests.helpers import require_bash_compatible_shell
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
    """Scenario mock returns Python that writes a file; chat() auto-runs it without a live API key.

    Two conversations back to back (messages reset between them):

    - User
      - message: "Write the word 'Washington' to a .txt file called file.txt. ..."
    - Assistant
      - code (python): open file.txt and write "Washington"
    - Computer
      - console output
    - Assistant
      - message: "The task is done."
    - User
      - message: "Read file.txt in the current directory and tell me what's in it."
    - Assistant
      - message: "Washington"
    """
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


_TOOL_CHAIN_PROMPT = (
    "Please demonstrate a tool chain: write to a file with python, modify it "
    "with shell, then recover from an error and report back."
)


def _assert_tool_chain_complete(interpreter, tmp_path, messages):
    """The tool chain ran python, shell, failing python, fixed python, then talked.

    step2.txt embeds step1.txt's content, proving the shell execution
    observed the python execution's filesystem state (cross-step, cross-
    language persistence) rather than each step running isolated. The
    failing step proves the loop survives execution errors; the fixed step
    proves it keeps executing afterwards. Full conversation:

    - User
      - message: "Please demonstrate a tool chain: ..."
    - Assistant
      - code (python): write "one" to step1.txt
    - Computer
      - console output (empty)
    - Assistant
      - code (shell): echo step1.txt content into step2.txt
    - Computer
      - console output
    - Assistant
      - code (python): print(undefined_name)
    - Computer
      - console output (NameError traceback)
    - Assistant
      - code (python): print("recovered")
    - Computer
      - console output
    - Assistant
      - message: "Tool chain complete."
    """
    assert (tmp_path / "step1.txt").read_text() == "one"
    assert (tmp_path / "step2.txt").read_text().strip() == "one-two"
    assert "Tool chain complete." in messages[-1]["content"]
    formats = [m.get("format") for m in messages if m.get("type") == "code"]
    assert formats == ["python", "shell", "python", "python"]
    console_text = "\n".join(
        m.get("content", "")
        for m in messages
        if m.get("type") == "console" and isinstance(m.get("content"), str)
    )
    assert "NameError" in console_text
    assert "recovered" in console_text


@pytest.mark.linux_ci
@pytest.mark.timeout(180)
def test_mock_llm_tool_chain(mock_llm_server, monkeypatch, tmp_path):
    """One tool-calling convo executes, fails, recovers, then talks.

    The mock server emits real OpenAI streaming tool_calls deltas (split
    across chunks); the run executes python, then shell, then a failing
    python step, then a fixed python step, then ends by talking — all
    through HTTP with no live API key. Conversation:

    - User
      - message: "Please demonstrate a tool chain: ..."
    - Assistant
      - tool_call execute(python): write "one" to step1.txt
    - Tool
      - result (empty output)
    - Assistant
      - tool_call execute(shell): echo step1.txt content into step2.txt
    - Tool
      - result
    - Assistant
      - tool_call execute(python): print(undefined_name)
    - Tool
      - result (NameError traceback)
    - Assistant
      - tool_call execute(python): print("recovered")
    - Tool
      - result
    - Assistant
      - message: "Tool chain complete."
    """
    require_bash_compatible_shell()
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    messages = interpreter.chat(
        _TOOL_CHAIN_PROMPT, display=False, stream=False, blocking=True
    )

    _assert_tool_chain_complete(interpreter, tmp_path, messages)


@pytest.mark.linux_ci
@pytest.mark.timeout(120)
def test_mock_llm_text_tool_chain(mock_llm_server, monkeypatch, tmp_path):
    """The same tool chain in code-block mode: fences, two languages, then talking.

    Conversation (code arrives as fenced text blocks, not tool_calls):

    - User
      - message: "Please demonstrate a tool chain: ..."
    - Assistant
      - message: ```python block writing "one" to step1.txt
    - Computer
      - console output (empty)
    - Assistant
      - message: ```shell block echoing step1.txt content into step2.txt
    - Computer
      - console output
    - Assistant
      - message: ```python block printing undefined_name
    - Computer
      - console output (NameError traceback)
    - Assistant
      - message: ```python block printing "recovered"
    - Computer
      - console output
    - Assistant
      - message: "Tool chain complete."
    """
    require_bash_compatible_shell()
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_interpreter(mock_llm_server, auto_run=True)

    messages = interpreter.chat(
        _TOOL_CHAIN_PROMPT, display=False, stream=False, blocking=True
    )

    _assert_tool_chain_complete(interpreter, tmp_path, messages)


@pytest.mark.linux_ci
@pytest.mark.timeout(120)
def test_mock_llm_tool_chain_after_prior_message(mock_llm_server, monkeypatch, tmp_path):
    """A tool chain started after an unrelated user message still begins at the python step.

    Turn counting is scoped to messages after the tool-chain prompt; the
    assistant turn from the earlier hello message must not shift the tool
    chain to turn 1. Conversation:

    - User
      - message: "Say hello."
    - Assistant
      - message: "Hello, World!"
    - User
      - message: "Please demonstrate a tool chain: ..."
    - Assistant
      - tool_call execute(python): write "one" to step1.txt
    - Tool
      - result (empty output)
    - Assistant
      - tool_call execute(shell): echo step1.txt content into step2.txt
    - Tool
      - result
    - Assistant
      - tool_call execute(python): print(undefined_name)
    - Tool
      - result (NameError traceback)
    - Assistant
      - tool_call execute(python): print("recovered")
    - Tool
      - result
    - Assistant
      - message: "Tool chain complete."
    """
    require_bash_compatible_shell()
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    interpreter.chat("Say hello.", display=False, stream=False, blocking=True)
    messages = interpreter.chat(
        _TOOL_CHAIN_PROMPT, display=False, stream=False, blocking=True
    )

    _assert_tool_chain_complete(interpreter, tmp_path, messages)


@pytest.mark.linux_ci
@pytest.mark.timeout(180)
def test_mock_llm_second_tool_chain_restarts_at_python(mock_llm_server, monkeypatch, tmp_path):
    """A second tool-chain run in one conversation restarts at the python step.

    Turn counting is scoped to the most recent tool-chain prompt; the
    completed first run's assistant turns must not shift the second run past
    its tool calls into an immediate "Tool chain complete." Conversation:

    - User
      - message: "Please demonstrate a tool chain: ..." (first run, full
        5-turn body as in test_mock_llm_tool_chain, ending "Tool chain
        complete.")
    - User
      - message: "Please demonstrate the tool chain again from the top."
    - Assistant
      - tool_call execute(python): write "one" to step1.txt
    - Tool
      - result (empty output)
    - Assistant
      - tool_call execute(shell): echo step1.txt content into step2.txt
    - Tool
      - result
    - Assistant
      - tool_call execute(python): print(undefined_name)
    - Tool
      - result (NameError traceback)
    - Assistant
      - tool_call execute(python): print("recovered")
    - Tool
      - result
    - Assistant
      - message: "Tool chain complete."
    """
    require_bash_compatible_shell()
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    interpreter.chat(
        _TOOL_CHAIN_PROMPT, display=False, stream=False, blocking=True
    )
    messages = interpreter.chat(
        "Please demonstrate the tool chain again from the top.",
        display=False,
        stream=False,
        blocking=True,
    )

    _assert_tool_chain_complete(interpreter, tmp_path, messages)


@pytest.mark.linux_ci
@pytest.mark.timeout(180)
def test_mock_llm_persistence_across_user_messages(
    mock_llm_server, monkeypatch, tmp_path
):
    """Session state defined before one user message is usable after the next.

    A single conversation holds two user messages on one interpreter (one
    kernel, one shell process): the first defines a python value and a shell
    value, the second uses both. Console outputs 42 and hello prove the state
    survived across user messages, not just consecutive loop turns.
    Conversation:

    - User
      - message: "Store values for later: set a python variable and a shell
        variable"
    - Assistant
      - tool_call execute(python): persist_num = 40 + 2
    - Tool
      - result
    - Assistant
      - tool_call execute(shell): export PERSIST_WORD=hello
    - Tool
      - result
    - Assistant
      - message: "values defined."
    - User
      - message: "Use the stored values: print both"
    - Assistant
      - tool_call execute(python): print(persist_num)
    - Tool
      - result (42)
    - Assistant
      - tool_call execute(shell): echo $PERSIST_WORD
    - Tool
      - result (hello)
    - Assistant
      - message: "values verified."
    """
    require_bash_compatible_shell()
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    interpreter.chat(
        "Store values for later: set a python variable and a shell variable",
        display=False,
        stream=False,
        blocking=True,
    )
    messages = interpreter.chat(
        "Use the stored values: print both",
        display=False,
        stream=False,
        blocking=True,
    )

    console_text = "\n".join(
        m.get("content", "")
        for m in messages
        if m.get("type") == "console" and isinstance(m.get("content"), str)
    )
    assert "42" in console_text
    assert "hello" in console_text
    assert "values verified." in messages[-1]["content"]


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
