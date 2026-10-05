import unittest
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
        self.system_message = "You are an assistant."
        self.custom_instructions = None
        self.offline = False
        self.os = False
        self.verbose = False
        self.loop = False
        self.loop_message = "Proceed."
        self.loop_breakers = ["The task is done."]
        self._stopped_retrying = False

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


def _bodiless_400():
    """A provider 400 that arrived with no body at all, as some gateways send."""
    return litellm.BadRequestError(
        message="OpenAIException - Error code: 400",
        model="m",
        llm_provider="openai",
        response=type("R", (), {"status_code": 400, "headers": {}})(),
    )


def test_bodiless_400_is_retried_once_without_asking_the_user(capsys, panels):
    """A 400 with an empty body is retried automatically instead of prompting.

    The OpenCode Go gateway rejects some otherwise valid requests with a bare
    "Error code: 400" and no body, and the identical request succeeds on a
    retry. Prompting "Retry? (y/a/n)" for that is pure friction -- the user
    learns nothing from the error and the answer is always the same -- so
    respond() retries once on its own and only surfaces the error if that
    fails too.
    """
    llm = _FakeLlm([_bodiless_400(), [{"type": "message", "content": "hi"}]])

    interpreter = _first_reply(llm)

    assert llm.calls == 2, "one failed attempt plus one automatic retry"
    assert interpreter._stopped_retrying is False, "a recovered turn is not a refusal"
    output = capsys.readouterr().out
    assert "Retry?" not in output, "the user must not be asked to retry an unexplainable error"


def test_bodiless_400_stops_after_one_retry(capsys, panels):
    """A persistently bodiless 400 is surfaced, not retried forever.

    The automatic retry is deliberately bounded. A gateway that rejects every
    attempt must produce an error panel and hand control back, otherwise the
    turn loops indefinitely against a provider that will never accept it.
    """
    llm = _FakeLlm([_bodiless_400(), _bodiless_400(), _bodiless_400()])

    interpreter = _FakeInterpreter(llm)
    list(respond(interpreter))

    assert llm.calls == 2, f"exactly one automatic retry, got {llm.calls} attempts"
    output = capsys.readouterr().out + "".join(panels)
    assert "Error code: 400" in output, "the user must be told the request was rejected"


def test_descriptive_400_is_not_auto_retried(capsys, panels):
    """A 400 that explains itself is a real complaint and stays actionable.

    When the body names a cause (a rejected field, a bad tool schema), retrying
    the identical request cannot help and would only hide the message the user
    needs. Only genuinely empty bodies are treated as transient.
    """
    error = litellm.BadRequestError(
        message="Error code: 400 - unsupported parameter: 'reasoning_effort'",
        model="m",
        llm_provider="openai",
        response=type("R", (), {"status_code": 400, "headers": {}})(),
    )
    llm = _FakeLlm([error])

    interpreter = _FakeInterpreter(llm)
    list(respond(interpreter))

    assert llm.calls == 1, "an explained error must not be retried automatically"
    output = capsys.readouterr().out + "".join(panels)
    assert "unsupported parameter" in output, "the explanation must reach the user"


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


def test_value_error_signals_the_interface_to_stop_not_retry(capsys, panels):
    """A config ValueError must end the turn in a way the interactive loop honours.

    Nothing is stored to interpreter.messages when no assistant text is produced,
    so the message list still looks like "a user message waiting to be served",
    and the terminal interface re-runs the same turn on its next iteration:
    with a permanent misconfiguration, that is a new "Configuration error"
    panel forever. The provider-error paths signal exit by setting
    interpreter._stopped_retrying, which the interface honours by popping the
    undelivered message and exiting. A turn that can never succeed must do the
    same; plain `return` was a real regression that ran this loop on a user.
    """
    error = ValueError(
        "The opencode_go/ model prefix requires an OpenCode Go API key. "
        "Set OPENCODE_GO_API_KEY, or set llm.api_key in your profile."
    )
    llm = _FakeLlm([error])
    interpreter = _FakeInterpreter(llm)

    list(respond(interpreter))

    assert interpreter._stopped_retrying is True, "a turn that cannot proceed must tell the interactive loop to stop"


def test_respond_resets_stale_stop_flag_at_turn_start(capsys, panels):
    """A stop flag left over from a previous failed turn must not kill the next one.

    The flag is consumed by the terminal interface after the turn it was set in.
    If a second turn starts while the flag is still True (the interface skipped
    its cleanup, or a library user drives turns manually), an entirely healthy
    turn would be reported as a refusal. respond() is the only writer, so it
    starts every turn with the flag False.
    """
    llm = _FakeLlm([[{"type": "message", "content": "all good"}]])
    interpreter = _FakeInterpreter(llm)
    interpreter._stopped_retrying = True  # stale, from the previous turn

    list(respond(interpreter))

    assert interpreter._stopped_retrying is False, "a healthy turn must clear a stale stop flag, not inherit it"


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


def _model_unavailable_error():
    """A gateway rejecting a model name that does not exist, as OpenCode Go does.

    Returned as a 400, which is the status for a permanent client-side mistake, but
    the wording ("Model is unavailable") is what the classification actually reads.
    """
    return litellm.BadRequestError(
        message="OpenAIException - Upstream request failed: Model is unavailable.",
        model="m",
        llm_provider="openai",
        response=type("R", (), {"status_code": 400, "headers": {}})(),
    )


def test_model_unavailable_is_not_treated_as_a_temporary_fault():
    """A model that does not exist stays unavailable; retrying cannot conjure it.

    "unavailable" used to be a bare substring match on the whole error, so this
    permanent condition was classified transient and retried forever. That is what
    turned a model typo into an unkillable loop: the retry kept the Rich Live
    display up, which reads stdin in raw mode, so Ctrl-C arrived as a keystroke
    rather than a signal and the user could not escape it.
    """
    assert respond_mod._is_temporary_provider_error(_model_unavailable_error()) is False


@pytest.mark.parametrize(
    "message",
    [
        # Genuinely transient phrasings that must keep retrying. "temporarily
        # unavailable" was already caught by the "temporarily" marker; the rest
        # only worked because of the bare "unavailable" match this change removes.
        "503 service unavailable",
        "Service unavailable, please retry",
        "The upstream provider is temporarily unavailable",
        # Plain transient signals, unaffected by the change.
        "Provider returned error 429, temporarily rate-limited",
        "upstream overloaded, try again later",
    ],
)
def test_genuinely_transient_failures_are_still_temporary(message):
    """Dropping the bare "unavailable" match must not cost us real retries.

    Each phrasing here would stop matching if the temporary-unavailability phrases
    were narrowed too far, silently turning a blip into an error the user has to
    acknowledge.
    """
    assert respond_mod._is_temporary_provider_error(Exception(message)) is True


def test_model_unavailable_is_surfaced_not_retried(capsys, panels):
    """The permanent model error reaches the user after one attempt.

    End-to-end guard on the same bug: the classifier is correct in isolation, but
    the loop must also stop calling the provider, since the infinite loop the user
    hit was the combination of the two.
    """
    llm = _FakeLlm([_model_unavailable_error()])

    interpreter = _FakeInterpreter(llm)
    list(respond(interpreter))

    assert llm.calls == 1, "a model that cannot be found must not be retried"
    output = capsys.readouterr().out + "".join(panels)
    # The panel body is truncated to the console width, so the useful assertion is
    # the title: a permanent fault renders as a red "Error", while the temporary
    # path would have rendered a yellow "Warning" and gone on to retry.
    assert "BadRequestError" in output, "the underlying error must reach the user"
    assert "Error" in output and "Warning" not in output, (
        "a permanent model error must be shown as an Error, not a temporary Warning"
    )


def test_temporary_retries_stop_at_the_cap(capsys, panels):
    """A provider that stays transiently broken is finally surfaced, not looped on.

    Regression: temporary_provider_error_retries was incremented, reset and drawn
    but never compared against a limit, so a provider that kept returning the same
    429 was retried indefinitely. The bodiless-400 counter beside it *was* bounded,
    and the comment above it claimed the retry machinery generally was -- which is
    what hid this. The cap must be a hard stop that hands control back.
    """
    error = _rate_limit_error()
    cap = respond_mod.MAX_TEMPORARY_PROVIDER_RETRIES
    # More than enough outcomes to loop forever if the cap were removed.
    llm = _FakeLlm([error] * (cap + 10))

    interpreter = _FakeInterpreter(llm)
    list(respond(interpreter))

    assert llm.calls == cap + 1, (
        f"expected {cap} automatic retries then a surfaced error, "
        f"got {llm.calls} attempts"
    )
    output = capsys.readouterr().out + "".join(panels)
    assert "429" in output, "the user must be told the provider kept failing"


def test_temporary_error_recovers_before_the_cap(capsys, panels):
    """Bounding the retries must not stop a normal blip from succeeding.

    The cap exists to stop a hopeless loop, not to ration retries: a rate limit
    that clears on the second attempt must still be retried automatically and
    produce the reply, exactly as before this change.
    """
    llm = _FakeLlm([_rate_limit_error(), [{"type": "message", "content": "hi"}]])

    interpreter = _first_reply(llm)

    assert llm.calls == 2, "one transient failure then a successful retry"
    assert interpreter._stopped_retrying is False, "a recovered turn is not a refusal"


class TestRetryPromptOffersExitOnlyWhenExitingHelps:
    """"Stop retrying" and "exit OI" must not be the same key.

    The prompt used to offer (y, a, n) with n meaning "stop", and n set
    _stopped_retrying -- which the interface treats as exit, ending the session.
    So the two cases a provider failure splits into were collapsed onto one key:

    - throttled or out of quota: fixed by refilling and waiting, so the user
      loses their session for no reason. No option just stopped.
    - wrong API key, model that does not exist: no in-session action helps, so
      leaving is the correct outcome.

    n now returns to the prompt in both cases, and e is offered only for the
    second. Whether the fault is recoverable keys off the error's nature, not off
    is_temporary_error -- exhausting the retry budget forces that flag False even
    for a throttle, so by the time the prompt is reached it no longer says which
    kind of failure this was.
    """

    def _run_prompt(self, error, choice, panels):
        """Drive respond() to the retry prompt and answer it with `choice`."""
        recorded = []

        def _prompt(text, options, **_):
            recorded.append((text, tuple(options)))
            return choice

        interpreter = _FakeInterpreter(_FakeLlm([error] * 20))
        respond_mod._stdin_is_interactive = lambda: True
        respond_mod.prompt_choice = _prompt
        try:
            list(respond(interpreter))
        finally:
            respond_mod._stdin_is_interactive = lambda: False
        assert recorded, "the retry prompt was never shown"
        return interpreter, recorded[0]

    def test_throttle_offers_no_exit_and_stays_in_the_conversation(self, panels):
        """A recoverable failure: n stops retrying and does not exit."""
        interpreter, (text, options) = self._run_prompt(_rate_limit_error(), "n", panels)

        assert "e" not in options, (
            "a throttle is fixed by refilling; quitting OI must not be on offer"
        )
        assert "return to the prompt" in text
        assert interpreter._stopped_retrying is False, (
            "n must not end the session; the user has to be able to carry on"
        )

    def test_permanent_error_offers_exit(self, panels):
        """An unrecoverable failure: e is available for leaving."""
        _, (text, options) = self._run_prompt(_auth_error(), "n", panels)

        assert "e" in options, "a wrong API key cannot be fixed in-session"
        assert "exit" in text

    def test_choosing_exit_still_ends_the_session(self, panels):
        """e keeps the old behaviour, so the permanent case is unchanged."""
        interpreter, _ = self._run_prompt(_auth_error(), "e", panels)

        assert interpreter._stopped_retrying is True, (
            "e must still ask the interface to exit"
        )

    def test_permanent_error_stays_in_conversation_on_n(self, panels):
        """n is uniformly non-fatal, including for a permanent fault.

        Uniform beats clever here: the user can always quit with Ctrl-C, and a
        prompt whose meaning shifts with the error is harder to trust than one
        that always means what it says.
        """
        interpreter, _ = self._run_prompt(_auth_error(), "n", panels)

        assert interpreter._stopped_retrying is False, "n means stop retrying, not quit"
