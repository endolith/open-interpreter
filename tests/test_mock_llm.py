import pytest

from interpreter import OpenInterpreter
from tests.helpers import require_bash_compatible_shell
from tests.support.mock_openai_server import MockOpenAIServer, stream_reply_chunks


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


_SENTENCE = "The quick brown fox jumps over the lazy dog."


def _request_carrying(requests: list[dict], marker: str, occurrence: int = 0) -> dict:
    """The first recorded request whose history contains an assistant step with `marker`.

    Selected by content rather than by index. The tool chain's final step is sent
    twice — once, and again after the model sees its traceback — so `requests[3]`
    and `requests[4]` both contain it. Indexing into that is fragile: adding or
    removing any earlier step shifts both, and the assertion would then be reading
    a different request entirely rather than failing.

    Which occurrence matters. The tool chain's failing step is sent twice: the first
    request carrying it also carries that step's own output, so it holds the
    traceback. The second is the follow-up the model sends after seeing it, and
    holds whatever the model did next — which is where the recovery output lands.
    """
    matches = [
        request
        for request in requests
        if any(
            message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and marker in message["content"]
            for message in request.get("messages", [])
        )
    ]
    assert len(matches) > occurrence, (
        f"expected at least {occurrence + 1} request(s) carrying {marker!r}, "
        f"found {len(matches)}"
    )
    return matches[occurrence]


def _last_user_text(request: dict) -> str:
    """The most recent user message text in a recorded request.

    The interpreter's last user message is the feedback turn: code output or a
    traceback. Asserting on that specific message is what makes "the feedback
    reached the model *now*" testable, as opposed to "it arrived at some point".
    """
    for message in reversed(request.get("messages", [])):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _user_texts(request: dict) -> list[str]:
    """User message contents from a recorded request, in the order sent."""
    return [
        m["content"]
        for m in request.get("messages", [])
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    ]


@pytest.mark.timeout(60)
def test_every_request_starts_with_the_system_message(mock_llm_server):
    """No request reaches the provider without the system message first.

    Nothing raises when the system message is dropped — the model simply stops
    being told what it is and how to behave, and every later request inherits
    the loss. Asserting on the request rather than the response is what makes
    this visible: a provider-side test can only see what came back.
    """
    interpreter = _mock_interpreter(mock_llm_server)

    interpreter.chat(
        "just the words hello, world", display=False, stream=False, blocking=True
    )

    assert mock_llm_server.requests, "expected the provider to be called at least once"
    for request in mock_llm_server.requests:
        messages = request.get("messages") or []
        assert messages[0]["role"] == "system", (
            "system message must be first, got "
            f"{[m.get('role') for m in messages]}"
        )
        assert messages[0].get("content"), "system message must not be empty"


@pytest.mark.timeout(60)
def test_a_plain_reply_ends_the_run_after_exactly_one_request(mock_llm_server):
    """A reply needing no execution produces one request, not a runaway loop.

    If the loop asks again after such a reply, every turn of a real session pays
    for it, and the extra requests all carry full history, so the cost compounds.
    """
    interpreter = _mock_interpreter(mock_llm_server)

    interpreter.chat(
        "just the words hello, world", display=False, stream=False, blocking=True
    )

    assert len(mock_llm_server.requests) == 1, (
        "a plain reply needs no follow-up, got "
        f"{len(mock_llm_server.requests)} requests"
    )


@pytest.mark.timeout(60)
def test_second_turn_carries_first_turn_history_in_order(mock_llm_server):
    """Turn two repeats turn one's user text verbatim, in sequence.

    Out-of-order or dropped history is the bug class where both halves of the
    system pass individually and the join is wrong: every message is
    well-formed, and only the sequence the model sees is off.
    """
    interpreter = _mock_interpreter(mock_llm_server)

    first = "hello, world please"
    second = "hello, world again"
    interpreter.chat(first, display=False, stream=False, blocking=True)
    interpreter.chat(second, display=False, stream=False, blocking=True)

    assert len(mock_llm_server.requests) >= 2
    sent = _user_texts(mock_llm_server.requests[-1])
    assert sent[0] == first
    assert sent[1] == second
    assert sent.index(first) < sent.index(second), "history must stay in turn order"


@pytest.mark.timeout(60)
def test_a_streamed_plain_reply_reassembles_into_one_message(mock_llm_server):
    """Word-sized deltas arriving over the wire become one clean message.

    Since #363 the mock streams prose as multiple content deltas, so this
    exercises real reassembly. The delta count is asserted as well as the final
    text: a correct final message is compatible with a dropped middle delta if
    something downstream compensates, so the text alone would not catch it.
    """
    assert len(stream_reply_chunks(_SENTENCE)) == 9
    interpreter = _mock_interpreter(mock_llm_server)

    messages = interpreter.chat(
        "say the quick brown fox", display=False, stream=False, blocking=True
    )

    assistant = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["content"] == _SENTENCE


@pytest.mark.timeout(60)
def test_a_single_delta_reply_reassembles_to_the_same_message(mock_llm_server):
    """The whole reply as one delta yields the identical message.

    OI only ever makes streaming requests (#358), so this is not a product mode
    but a fake-side regression guard: if the splitter regressed to [content] or
    a future fake stopped splitting, the streamed test above would still pass
    while only one arrival shape was ever exercised.
    """
    interpreter = _mock_interpreter(mock_llm_server)
    mock_llm_server.single_delta = True

    messages = interpreter.chat(
        "say the quick brown fox", display=False, stream=False, blocking=True
    )

    assert messages[-1]["content"] == _SENTENCE


@pytest.mark.timeout(60)
def test_a_traceback_reaches_the_provider_on_the_next_request(
    mock_llm_server, monkeypatch, tmp_path
):
    """A failed execution's traceback is sent back to the model, not swallowed.

    This is the residual gap #353 identified after its own tests were ported:
    the mock's tool-chain machine advances by counting assistant turns, so it never
    looks at error text. A regression that stops feeding tracebacks to the
    provider would therefore still let every other test in this file pass, and
    the model would silently never learn why its code failed.
    """
    require_bash_compatible_shell()
    # The tool-chain scenario writes step1.txt/step2.txt into the cwd, so keep
    # them out of the checkout the way the sibling tool-chain tests do.
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_interpreter(mock_llm_server, auto_run=True)

    interpreter.chat(
        _TOOL_CHAIN_PROMPT,
        display=False,
        stream=False,
        blocking=True,
    )

    assert len(mock_llm_server.requests) >= 2
    # The request immediately following the failing step must carry the failure,
    # not merely some later request. Searching every later request would pass
    # even if the feedback were delayed by a turn, which is the same regression
    # seen from the other direction.
    follow_up = _last_user_text(
        _request_carrying(mock_llm_server.requests, "undefined_name")
    )
    assert "Traceback (most recent call last)" in follow_up, (
        "the request right after the failing step carried no traceback header"
    )
    assert "NameError" in follow_up, (
        "the request right after the failing step did not name the exception"
    )


@pytest.mark.timeout(60)
def test_execution_output_reaches_the_provider_on_the_next_request(
    mock_llm_server, monkeypatch, tmp_path
):
    """Console output from executed code is fed back into the following request.

    Partially covered on main by the write-to-file test, but only indirectly: it
    fails under a regression in this path by running away until its step
    timeout, because the mock only stops looping once it sees output arrive. The
    failure class is caught, but nothing asserts that the output is actually in
    the request, so assert it directly.
    """
    require_bash_compatible_shell()
    # The tool-chain scenario writes step1.txt/step2.txt into the cwd, so keep
    # them out of the checkout the way the sibling tool-chain tests do.
    monkeypatch.chdir(tmp_path)
    interpreter = _mock_interpreter(mock_llm_server, auto_run=True)

    interpreter.chat(
        _TOOL_CHAIN_PROMPT,
        display=False,
        stream=False,
        blocking=True,
    )

    # Assert the actual stdout, not the fixed "Code output:" label, which every
    # feedback message carries regardless of whether output arrived.
    follow_up = _last_user_text(
        _request_carrying(mock_llm_server.requests, "undefined_name", occurrence=1)
    )
    assert follow_up.startswith("Code output:"), (
        "the request right after execution did not carry a code-output feedback"
    )
    assert "recovered" in follow_up, (
        "the request right after execution did not carry the actual stdout"
    )
