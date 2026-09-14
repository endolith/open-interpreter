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
    """The errand ran python, shell, failing python, fixed python, then talked.

    step2.txt embeds step1.txt's content, proving the shell execution
    observed the python execution's filesystem state (cross-step, cross-
    language persistence) rather than each step running isolated. The
    failing step proves the loop survives execution errors; the fixed step
    proves it keeps executing afterwards.
    """
    assert (tmp_path / "step1.txt").read_text() == "one"
    assert (tmp_path / "step2.txt").read_text().strip() == "one-two"
    assert "Errand complete." in messages[-1]["content"]
    formats = [m.get("format") for m in messages if m.get("type") == "code"]
    assert formats == ["python", "shell", "python", "python"]
    console_text = "\n".join(
        m.get("content", "")
        for m in messages
        if m.get("type") == "console" and isinstance(m.get("content"), str)
    )
    assert "NameError" in console_text
    assert "recovered" in console_text


@pytest.mark.timeout(180)
def test_mock_llm_tool_call_errand(mock_llm_server, monkeypatch, tmp_path):
    """One tool-calling convo executes, fails, recovers, then talks.

    The mock server emits real OpenAI streaming tool_calls deltas (split
    across chunks); the run executes python, then shell, then a failing
    python step, then a fixed python step, then ends by talking — all
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


@pytest.mark.timeout(180)
def test_mock_llm_second_errand_restarts_at_python(mock_llm_server, monkeypatch, tmp_path):
    """A second errand in one conversation restarts at the python step.

    Turn counting is scoped to the most recent errand prompt; the completed
    first errand's assistant turns must not shift the second errand past its
    tool calls into an immediate "Errand complete."
    """
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    interpreter.chat(
        _ERRAND_PROMPT, display=False, stream=False, blocking=True
    )
    messages = interpreter.chat(
        "Please run this errand again from the top.",
        display=False,
        stream=False,
        blocking=True,
    )

    _assert_errand_complete(interpreter, tmp_path, messages)


@pytest.mark.timeout(180)
def test_mock_llm_split_persistence_across_chats(
    mock_llm_server, monkeypatch, tmp_path
):
    """Session state defined in one chat is usable in the next.

    Two chats share one interpreter (one kernel, one shell process), split by
    a genuine user message: the first defines a python value and a shell
    value, the second uses both. Console outputs 42 and hello prove the state
    survived across chat() calls, not just consecutive loop turns.
    """
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    interpreter.chat(
        "persistence check part one: define a python value and a shell value",
        display=False,
        stream=False,
        blocking=True,
    )
    messages = interpreter.chat(
        "persistence check part two: print both values",
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


def _console_text(messages) -> str:
    """All console output in a conversation, joined."""
    return "\n".join(
        m.get("content", "")
        for m in messages
        if m.get("type") == "console" and isinstance(m.get("content"), str)
    )


def _code_contents(messages) -> list:
    """The (format, content) of every code message, in order."""
    return [(m["format"], m["content"]) for m in messages if m.get("type") == "code"]


@pytest.mark.timeout(120)
def test_mock_llm_tool_call_name_only_opener(mock_llm_server, monkeypatch, tmp_path):
    """A call announced with empty arguments executes once its arguments arrive.

    OpenAI's first tool_calls delta carries the id and function name with
    arguments "", and the JSON follows in later deltas. The client must
    neither execute nor discard the call on the empty opener; it must
    assemble the later argument text into one execution.
    """
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    messages = interpreter.chat("name-first opener please", display=False, stream=False, blocking=True)

    assert _code_contents(messages) == [("python", 'print("opener ok")')]
    assert "opener ok" in _console_text(messages)
    assert messages[-1]["content"] == "opener done."


@pytest.mark.timeout(120)
def test_mock_llm_tool_call_cut_mid_token_and_mid_escape(mock_llm_server, monkeypatch, tmp_path):
    """Argument deltas cut inside "python" and inside a \\uXXXX escape still run correctly.

    Providers cut argument JSON at arbitrary byte offsets. A cut inside the
    language token must not latch a partial language, and a cut inside a
    unicode escape must not corrupt the non-ASCII characters it encodes, so
    the executed code and its output must match the scenario exactly.
    """
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    messages = interpreter.chat("unicode cut please", display=False, stream=False, blocking=True)

    assert _code_contents(messages) == [("python", 'print("café ✓")')]
    assert "café ✓" in _console_text(messages)
    assert messages[-1]["content"] == "unicode done."


@pytest.mark.timeout(120)
def test_mock_llm_parallel_calls_in_one_delta_known_defect(mock_llm_server, monkeypatch, tmp_path):
    """KNOWN DEFECT: a second tool call in the same delta is silently dropped.

    OpenAI may answer with two tool_calls entries (index 0 and 1) in one
    delta. run_tool_calling_llm reads only entry [0] of each delta, so the
    index-1 call never reaches parsing: the first call executes, its output
    is the only console output, and no error or warning is raised.

    Correct behavior is to execute both calls, in order, yielding two code
    messages and both outputs. This test pins the current behavior so that
    a fix is noticed; flip the second-call assertions when parallel tool
    calls are supported.
    """
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    messages = interpreter.chat("two calls at once please", display=False, stream=False, blocking=True)

    assert _code_contents(messages) == [("python", 'print("first of two")')]
    assert "first of two" in _console_text(messages)
    assert "second-of-two" not in _console_text(messages)
    assert messages[-1]["content"] == "pair done."


@pytest.mark.timeout(120)
def test_mock_llm_parallel_calls_staggered_known_defect(mock_llm_server, monkeypatch, tmp_path):
    """KNOWN DEFECT: a second tool call arriving in a later delta is silently dropped.

    When the index-1 entry comes in a delta of its own, run_tool_calling_llm
    takes entry [0] of that delta, which is the second call, and merge_deltas
    appends its arguments onto the first call's accumulated arguments. The
    result is two JSON objects back to back, which parse_partial_json
    rejects, so the second call is discarded without a trace. The first
    call still executes because its arguments were already complete.

    Correct behavior is to track calls by index and execute both. This test
    pins the current behavior; flip the second-call assertions when
    parallel tool calls are supported.
    """
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    messages = interpreter.chat("two calls staggered please", display=False, stream=False, blocking=True)

    assert _code_contents(messages) == [("python", 'print("first of two")')]
    assert "first of two" in _console_text(messages)
    assert "second-of-two" not in _console_text(messages)
    assert messages[-1]["content"] == "staggered done."


@pytest.mark.timeout(120)
def test_mock_llm_content_alongside_tool_call_known_defect(mock_llm_server, monkeypatch, tmp_path):
    """KNOWN DEFECT: text sent in the same delta as a tool call is dropped.

    A streaming delta may carry both content and tool_calls. On seeing
    tool_calls, run_tool_calling_llm replaces the whole delta with a
    function_call dict, so the content key is gone before the message
    branch runs: the narration never becomes an assistant message, while
    the call itself executes normally.

    Correct behavior is to keep the content and yield it as a message
    before the code. This test pins the current behavior; flip the
    narration assertion when mixed deltas are handled.
    """
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    messages = interpreter.chat("narrated call please", display=False, stream=False, blocking=True)

    assert _code_contents(messages) == [("python", 'print("narrated ok")')]
    assert "narrated ok" in _console_text(messages)
    assert not any("Running it now." in (m.get("content") or "") for m in messages if m["role"] == "assistant")
    assert messages[-1]["content"] == "narrated done."


@pytest.mark.timeout(120)
def test_mock_llm_auth_accepts_judge_trailer_after_tool_call(mock_llm_server, monkeypatch, tmp_path):
    """A <safe> verdict streamed after a tool call satisfies the authentication guard.

    With INTERPRETER_REQUIRE_AUTHENTICATION set, run_tool_calling_llm raises
    when a function call arrives without a judge review. The review is
    plain content after the call; it must be recognized as the verdict
    rather than as a message, the code must still execute, and the review
    text must stay ephemeral (never stored as an assistant message).
    """
    monkeypatch.setenv("INTERPRETER_REQUIRE_AUTHENTICATION", "true")
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_tool_interpreter(mock_llm_server)

    messages = interpreter.chat("judged call please", display=False, stream=False, blocking=True)

    assert _code_contents(messages) == [("python", 'print("judged ok")')]
    assert "judged ok" in _console_text(messages)
    assert not any("Prints a constant." in (m.get("content") or "") for m in messages)
    assert messages[-1]["content"] == "judged done."


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
