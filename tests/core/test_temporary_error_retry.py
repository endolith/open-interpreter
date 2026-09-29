import io

import litellm
import pytest
from rich.console import Console

import interpreter.core.respond as respond_mod
from interpreter.core.respond import respond


class _FakeLlm:
    """Replays a scripted sequence of run() outcomes; an exception raises, a list yields."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def run(self, messages):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        yield from outcome


class _FakeInterpreter:
    """The small slice of Interpreter that respond() touches around an LLM call."""

    def __init__(self, llm):
        self.llm = llm
        self.messages = [{"role": "user", "type": "message", "content": "hi"}]
        self.offline = False
        self.os = False
        self.verbose = False

    def display_message(self, message):
        pass


def _rate_limit_error():
    return litellm.exceptions.RateLimitError(
        message="Provider returned error 429, temporarily rate-limited",
        llm_provider="openrouter",
        model="test/model",
    )


def _auth_error():
    return litellm.exceptions.AuthenticationError(
        message="Invalid API key provided",
        llm_provider="openrouter",
        model="test/model",
    )


@pytest.fixture
def panels(monkeypatch):
    """Record the Rich error panels respond() prints, rendered to plain text.

    Rich passes a Panel object to rich_print; rendering it through a Console is
    the only way to see the title/body the user actually sees, since str(Panel)
    is just a repr.
    """
    recorded = []

    def _render_to_text(renderable):
        console = Console(file=io.StringIO(), width=80, legacy_windows=False, force_terminal=False)
        console.print(renderable)
        recorded.append(console.file.getvalue())

    monkeypatch.setattr(respond_mod, "rich_print", _render_to_text)
    monkeypatch.setattr(respond_mod, "assemble_system_message", lambda interpreter: "system")
    monkeypatch.setattr(respond_mod, "_stdin_is_interactive", lambda: False)
    monkeypatch.setattr(respond_mod.time, "sleep", lambda seconds: None)
    return recorded


def _first_reply(llm):
    """Drive respond() until the LLM finally yields assistant text, then stop."""
    interpreter = _FakeInterpreter(llm)
    stream = respond(interpreter)
    try:
        for chunk in stream:
            if chunk.get("type") == "message" and "content" in chunk:
                break
    finally:
        stream.close()
    return interpreter


def _assert_retry_status_is_terminated(output):
    """The final in-place retry line must be newline-terminated before anything else draws.

    Splits on the last carriage return, which is the in-place update boundary between
    repeated retries, and requires the final segment to carry the status and end in a
    newline. This is independent of how many dots the spinner happened to draw.
    """
    final_segment = output.rsplit("\r", 1)[-1]
    assert "Temporary upstream provider error" in final_segment
    assert final_segment.endswith("\n"), (
        "the retry status line must end in a newline; otherwise the next renderer "
        "(e.g. the Rich Live Thinking panel) draws on the same row and is redrawn"
    )


def test_retry_status_is_terminated_before_streaming_reply(capsys, panels):
    """A successful retry must not stream its reply onto the retry status row.

    Regression: the retry status is written with a carriage return and no trailing
    newline so repeated retries update in place. When the retry then succeeded, the
    terminal began the Rich Live "Thinking" panel at that same cursor position, so the
    panel's top border was appended to the status text and immediately redrawn -- a
    doubled, wrapped Thinking pane.
    """
    llm = _FakeLlm([_rate_limit_error(), [{"type": "message", "content": "hi"}]])

    _first_reply(llm)

    output = capsys.readouterr().out
    assert "Temporary upstream provider error" in output
    _assert_retry_status_is_terminated(output)
    assert output.endswith("\n")


def test_repeated_identical_retries_update_in_place_then_terminate(capsys, panels):
    """Two identical temporary failures reuse one status line, terminated exactly once.

    Deduplication exists so a provider that keeps returning the same 429 does not
    reprint a full error panel on every attempt. This guards that the status is still
    updated in place (no accumulated blank lines) while the success path closes it off.
    """
    error = _rate_limit_error()
    llm = _FakeLlm([error, error, [{"type": "message", "content": "hi"}]])

    _first_reply(llm)

    output = capsys.readouterr().out
    assert output.count("Temporary upstream provider error") == 2
    assert len(panels) == 1, "the repeated error must not reprint its panel"
    _assert_retry_status_is_terminated(output)
    assert not output.endswith("\n\n"), "the status line must not be closed more than once"


def test_new_error_panel_after_retry_starts_on_a_fresh_line(capsys, panels):
    """A different error after a retry is still reported, and closes off the status line.

    Guards the dedup boundary: only a repeat of the *same* temporary error may be
    collapsed into the in-place status, so a subsequent, different error must still
    render its own panel rather than being swallowed as a duplicate.
    """
    llm = _FakeLlm([_rate_limit_error(), _auth_error()])

    list(respond(_FakeInterpreter(llm)))

    output = capsys.readouterr().out
    _assert_retry_status_is_terminated(output)
    assert len(panels) == 2, "both the 429 and the auth error should be reported"


def test_value_error_is_shown_without_traceback_or_retry_prompt(capsys, panels):
    """A required-setting ValueError is rendered as its own message, cleanly.

    OI raises ValueError for missing setup (e.g. the opencode_go/ prefix without
    OPENCODE_GO_API_KEY). Printing the exception directly would unfold a long
    traceback at the user; worse, the message says "requires an ... API key",
    which the generic auth handler matches on "api key" and answers with advice
    about resetting OPENAI_API_KEY -- the wrong credential entirely.
    """
    error = ValueError(
        "The opencode_go/ model prefix requires an OpenCode Go API key. "
        "Set OPENCODE_GO_API_KEY, or set llm.api_key in your profile."
    )
    llm = _FakeLlm([error])

    try:
        list(respond(_FakeInterpreter(llm)))
    except ValueError:
        pytest.fail("ValueError from a required-setting check must not escape respond()")

    output = capsys.readouterr().out + "".join(panels)
    assert "Configuration error" in output, "the message should render in its own panel"
    assert "OPENCODE_GO_API_KEY" in output, "the actionable instruction should survive rendering"
    assert "Traceback" not in output, "misconfiguration is not a crash; no traceback"
    assert "There might be an issue with your API key(s)" not in output, (
        "the generic OPENAI_API_KEY auth advice must not show for a known misconfiguration"
    )
    assert "OPENAI_API_KEY" not in output.replace("OPENCODE_GO_API_KEY", ""), (
        "generic OpenAI reset instructions have nothing to do with this error"
    )


def test_value_error_quits_instead_of_prompting_for_retry(capsys, panels):
    """A config ValueError terminates the turn; retrying cannot fix missing setup.

    Retrying a provider error makes sense because the upstream outage may clear.
    No amount of retries creates an API key, so prompting y/a/n (or looping in
    stdin mode) would ask the user to repeat a known-impossible action.
    """
    error = ValueError(
        "The opencode_go/ model prefix requires an OpenCode Go API key. "
        "Set OPENCODE_GO_API_KEY, or set llm.api_key in your profile."
    )
    # respond() calls _stdin_is_interactive() only when it wants to prompt for
    # retry; a third call would prove retry prompting happened. Record calls
    # instead so the test fails loudly if a prompt is attempted.
    calls = []
    monkeypatched = panels
    original_is_interactive = respond_mod._stdin_is_interactive
    respond_mod._stdin_is_interactive = lambda: calls.append(True) or False  # never interactive, just observed

    try:
        list(respond(_FakeInterpreter(_FakeLlm([error]))))
    finally:
        respond_mod._stdin_is_interactive = original_is_interactive

    output = capsys.readouterr().out
    assert "Retry?" not in output, "a missing key is not retryable"
    assert not calls, "respond() must not consult stdin for a retry prompt here"
