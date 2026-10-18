import itertools
from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.respond import _error_body, display_provider_error, respond


class FakeLanguage:
    name = "Python"
    file_extension = "py"


def _code_interpreter(*, language="python", code="1+1"):
    terminal = SimpleNamespace(
        languages=[FakeLanguage()],
        get_language=lambda lang: FakeLanguage() if lang == "python" else None,
    )

    def run(lang, code_to_run, **run_kwargs):
        yield {"type": "console", "format": "output", "content": "42"}

    computer = SimpleNamespace(
        terminal=terminal,
        import_computer_api=False,
        system_message="",
        run=run,
        verbose=False,
        debug=False,
        emit_images=False,
        max_output=2800,
        save_skills=True,
        to_dict=lambda: {},
    )

    return SimpleNamespace(
        system_message="You are helpful.",
        custom_instructions="",
        messages=[
            {"role": "assistant", "type": "code", "format": language, "content": code}
        ],
        computer=computer,
        llm=SimpleNamespace(run=lambda msgs: iter([]), supports_vision=False, max_budget=0),
        verbose=False,
        debug=False,
        auto_run=True,
        loop=False,
        loop_message="continue",
        loop_breakers=[],
        sync_computer=False,
        offline=True,
        os=False,
        display_message=mock.Mock(),
        max_output=2800,
    )


def test_unsupported_language_yields_console_output():
    """Unknown language names produce a console error instead of executing code."""
    interpreter = _code_interpreter(language="brainfuck", code="++")
    chunks = list(itertools.islice(respond(interpreter), 3))
    outputs = [c for c in chunks if c.get("format") == "output"]
    assert outputs
    assert "disabled or not supported" in outputs[0]["content"]


def test_text_language_converted_to_assistant_message():
    """Code blocks with language 'text' are stored as plain assistant messages, not executed."""
    interpreter = _code_interpreter(language="text", code="notes here")
    list(itertools.islice(respond(interpreter), 1))
    assert interpreter.messages[-1]["type"] == "message"
    assert "notes here" in interpreter.messages[-1]["content"]


def test_functions_execute_hallucination_parsed():
    """LLM hallucinations like functions.execute({...}) are parsed into real executable code."""
    code = 'functions.execute({"language": "python", "code": "7*6"})'
    interpreter = _code_interpreter(code=code)
    gen = respond(interpreter)
    list(itertools.islice(gen, 3))
    assert interpreter.messages[0]["content"] == "7*6"
    assert interpreter.messages[0]["format"] == "python"


def test_executeexecute_suffix_stripped():
    """A trailing 'executeexecute' artifact from model output is removed before code runs."""
    interpreter = _code_interpreter(code="print(1)executeexecute")
    list(itertools.islice(respond(interpreter), 3))
    assert interpreter.messages[0]["content"] == "print(1)"


def test_json_code_block_parsed():
    """JSON-shaped code content is parsed to extract language and code fields for execution."""
    interpreter = _code_interpreter(code='{"language": "python", "code": "8*8"}')
    list(itertools.islice(respond(interpreter), 3))
    assert interpreter.messages[0]["content"] == "8*8"


def test_llm_run_when_last_message_not_code():
    """When the last message is not code, respond delegates to llm.run instead of executing."""
    interpreter = _code_interpreter()
    interpreter.messages = [{"role": "user", "type": "message", "content": "hi"}]
    interpreter.llm.run = lambda msgs: iter([{"type": "message", "content": "hello"}])
    chunks = list(respond(interpreter))
    assert {"role": "assistant", "type": "message", "content": "hello"} in chunks


def _message_interpreter():
    """An interpreter whose last message is plain text (no code to execute)."""
    interpreter = _code_interpreter()
    interpreter.messages = [{"role": "user", "type": "message", "content": "hi"}]
    return interpreter


def test_respond_assembles_full_system_message():
    """respond() builds the system message from the base, language-specific
    messages, custom instructions, and the computer API message."""
    interpreter = _message_interpreter()
    interpreter.computer.import_computer_api = True
    interpreter.computer.system_message = "COMPUTER MSG"
    interpreter.custom_instructions = "CUSTOM MSG"

    class LangWithSystemMessage:
        name = "Python"
        file_extension = "py"
        system_message = "LANG MSG"

    interpreter.computer.terminal.languages = [LangWithSystemMessage()]
    captured = {}
    interpreter.llm.run = lambda msgs: (captured.update(msgs=msgs), iter([]))[1]

    list(respond(interpreter))

    content = captured["msgs"][0]["content"]
    assert content.startswith("You are helpful.")
    assert "LANG MSG" in content
    assert "CUSTOM MSG" in content
    assert "COMPUTER MSG" in content


def _panel_text(mock_rich_print):
    """Return the text a Panel was asked to render.

    str(Panel) is just the object repr, so assertions reach into the Markdown
    renderable the Panel wraps and read its source text -- that is what the
    user ends up looking at once Rich renders it.
    """
    panel = mock_rich_print.call_args[0][0]
    renderable = panel.renderable
    return str(getattr(renderable, "markup", renderable))


def test_respond_handles_budget_exceeded():
    """respond() reports the session/max budget and stops on BudgetExceededError.

    max_budget lives on interpreter.llm, not on the interpreter itself (the CLI
    flag and the profile loader both write interpreter.llm.max_budget). The
    handler read interpreter.max_budget, so reaching the budget raised
    AttributeError instead of showing the message. The fake interpreter now
    carries max_budget where the real one does, so the message must render.
    """
    import litellm

    interpreter = _message_interpreter()
    interpreter.llm.max_budget = 5

    def run(msgs):
        raise litellm.exceptions.BudgetExceededError(current_cost=0, max_budget=5)

    interpreter.llm.run = run
    list(respond(interpreter))

    shown = interpreter.display_message.call_args[0][0]
    assert "Max budget exceeded" in shown
    assert "5" in shown  # the limit is reported, not swallowed by an AttributeError


def test_respond_shows_the_providers_own_auth_error():
    """An auth error is shown verbatim and re-raised, not rewritten as OpenAI advice.

    The provider's message names the provider and the credential that failed,
    which is what the reader needs. Substituting generic "issue with your API
    key" text lost that, and pointed users at OPENAI_API_KEY regardless of
    which provider had actually refused the request.
    """
    interpreter = _message_interpreter()
    interpreter.offline = False

    def run(msgs):
        raise Exception("litellm.AuthenticationError: anthropic api key sk-ant-... invalid")

    interpreter.llm.run = run
    with mock.patch("interpreter.core.respond.rich_print") as rich:
        with pytest.raises(Exception, match="anthropic api key"):
            list(respond(interpreter))

    assert "anthropic api key" in _panel_text(rich)


def test_respond_does_not_swallow_a_rate_limit_error():
    """A quota failure reaches the caller instead of ending the session silently.

    The old handler printed a billing paragraph and did not re-raise, so a rate
    limit ended the turn with no output and no exception -- the user saw a
    message and then nothing else happened.
    """
    import litellm

    interpreter = _message_interpreter()
    interpreter.offline = False

    def run(msgs):
        raise litellm.exceptions.RateLimitError(
            "You exceeded your current quota", "openai", "gpt-4o"
        )

    interpreter.llm.run = run
    with mock.patch("interpreter.core.respond.rich_print") as rich:
        with pytest.raises(litellm.exceptions.RateLimitError):
            list(respond(interpreter))

    assert "exceeded your current quota" in _panel_text(rich)


def test_respond_reports_a_mistyped_model_as_a_mistyped_model():
    """A wrong model name is reported as a wrong model name, not a billing problem.

    The provider says "The model 'gpt-4oo' does not exist or you do not have
    access to it". The old handler looked for the literal substring "model does
    not exist", which does not appear there, so it fell through to its default
    branch and told the user to add a payment method.
    """
    interpreter = _message_interpreter()
    interpreter.offline = False
    interpreter.llm.model = "gpt-4oo"

    def run(msgs):
        raise Exception(
            "litellm.BadRequestError: The model `gpt-4oo` does not exist "
            "or you do not have access to it."
        )

    interpreter.llm.run = run
    with mock.patch("interpreter.core.respond.rich_print") as rich:
        with pytest.raises(Exception, match="does not exist"):
            list(respond(interpreter))

    rendered = _panel_text(rich)
    assert "does not exist" in rendered
    assert "payment method" not in rendered
    assert "billing" not in rendered.lower()


def test_respond_error_does_not_prompt_or_change_the_model():
    """An access error shows the provider's message and re-raises, with no prompt.

    The old handler asked whether to switch to Open Interpreter's hosted `i`
    model and blocked on input(), which also broke headless runs. Nothing is
    asked and the configured model is left alone.
    """
    interpreter = _message_interpreter()
    interpreter.offline = False
    interpreter.llm.model = "gpt-4o"

    def run(msgs):
        raise Exception("You do not have access to this model")

    interpreter.llm.run = run
    with mock.patch("builtins.input") as prompt:
        with mock.patch("interpreter.core.respond.rich_print"):
            with pytest.raises(Exception, match="have access"):
                list(respond(interpreter))

    prompt.assert_not_called()
    assert interpreter.llm.model == "gpt-4o"


def test_display_provider_error_names_the_exception_type_when_there_is_no_message():
    """A provider that raises without a message still produces a usable panel.

    Empty-string errors are rare but real; an empty box tells the user nothing
    about what failed.
    """
    with mock.patch("interpreter.core.respond.rich_print") as rich:
        display_provider_error(Exception(""))

    assert "Exception" in _panel_text(rich)


def test_display_provider_error_unwraps_an_html_body():
    """An HTML error page is converted to readable text instead of shown as markup.

    Gateways commonly answer a failed request with an HTML page. Rendered as
    Markdown, the markup is noise that buries the status line the user needs.
    """
    error = (
        "litellm.APIError: OpenrouterException - <!DOCTYPE html><html><body>"
        "<h1>502 Bad Gateway</h1></body></html>"
    )
    rendered = str(_error_body(error))

    assert "502 Bad Gateway" in rendered
    assert "OpenrouterException" in rendered
    assert "<!DOCTYPE" not in rendered


def test_display_provider_error_leaves_plain_text_alone():
    """A normal error message is passed through untouched."""
    error = "litellm.BadRequestError: unsupported parameter: 'reasoning_effort'"
    assert _error_body(error) == error


def test_respond_offline_errors_are_re_raised():
    """Offline sessions re-raise errors rather than swallowing them."""
    interpreter = _message_interpreter()
    interpreter.llm.model = "gpt-4o"

    def run(msgs):
        raise Exception("You do not have access to this model")

    interpreter.llm.run = run
    with mock.patch("interpreter.core.respond.rich_print"):
        with pytest.raises(Exception, match="have access"):
            list(respond(interpreter))


def test_respond_rewrites_import_computer():
    """Python code is stripped of `import computer` when the computer API is
    enabled."""
    interpreter = _code_interpreter(code="import computer\nprint('hi')")
    interpreter.computer.import_computer_api = True
    captured = []

    def run(lang, code, stream=False):
        captured.append(code)
        return iter([{"type": "console", "format": "output", "content": "ok"}])

    interpreter.computer.run = run
    list(itertools.islice(respond(interpreter), 3))

    assert captured
    assert "import computer\n" not in captured[0]
    assert "print('hi')" in captured[0]


def test_respond_parses_bare_json_language_dict():
    """Bare {language: ..., code: ...} dicts are parsed like JSON code blocks."""
    interpreter = _code_interpreter(code='{language: "python", code: "9*9"}')
    list(itertools.islice(respond(interpreter), 3))
    assert interpreter.messages[0]["content"] == "9*9"


def test_respond_yields_traceback_when_code_runner_fails():
    """A failing code runner yields the traceback as console output."""
    interpreter = _code_interpreter()

    def run(lang, code, stream=False):
        raise RuntimeError("nope")

    interpreter.computer.run = run
    chunks = list(itertools.islice(respond(interpreter), 4))
    outputs = [c for c in chunks if c.get("format") == "output"]
    assert any("RuntimeError" in c["content"] for c in outputs)


def test_respond_loop_message_is_inserted():
    """respond() appends the loop message and re-runs the LLM until a loop
    breaker appears."""
    interpreter = _code_interpreter()
    interpreter.messages = [
        {"role": "assistant", "type": "message", "content": "partial answer"}
    ]
    interpreter.loop = True
    interpreter.loop_message = "Proceed."
    interpreter.loop_breakers = ["The task is complete"]
    captured = {}

    def make_run():
        calls = {"n": 0}

        def run(msgs):
            captured["msgs"] = msgs
            calls["n"] += 1
            if calls["n"] == 1:
                # First response has no loop breaker, so respond() inserts the
                # loop message for the next turn.
                interpreter.messages.append(
                    {"role": "assistant", "type": "message", "content": "still working"}
                )
            else:
                # The second response completes the task.
                interpreter.messages.append(
                    {
                        "role": "assistant",
                        "type": "message",
                        "content": "The task is complete",
                    }
                )
            return iter([])

        return run

    interpreter.llm.run = make_run()
    list(respond(interpreter))

    assert captured["msgs"][-1]["content"] == "Proceed."


def test_respond_os_mode_expands_loop_message():
    """In OS mode the loop message asks for a verification screenshot."""
    interpreter = _code_interpreter()
    interpreter.messages = [
        {"role": "assistant", "type": "message", "content": "partial"}
    ]
    interpreter.loop = True
    interpreter.loop_message = (
        "If the entire task I asked for is done, stop."
    )
    interpreter.loop_breakers = ["stop"]
    interpreter.os = True
    captured = {}

    def make_run():
        calls = {"n": 0}

        def run(msgs):
            captured["msgs"] = msgs
            calls["n"] += 1
            if calls["n"] == 1:
                interpreter.messages.append(
                    {"role": "assistant", "type": "message", "content": "still working"}
                )
            else:
                interpreter.messages.append(
                    {
                        "role": "assistant",
                        "type": "message",
                        "content": "The task is done, stop",
                    }
                )
            return iter([])

        return run

    interpreter.llm.run = make_run()
    list(respond(interpreter))

    assert "take a screenshot to verify" in captured["msgs"][-1]["content"]


def test_respond_requires_messages():
    """respond() raises AssertionError when called with an empty message list."""
    interpreter = _code_interpreter()
    interpreter.messages = []
    with pytest.raises(AssertionError, match="User message was not passed"):
        next(respond(interpreter))


def test_error_body_without_prefix_returns_converted_body():
    """An error that starts with markup converts without a prefix line.

    When there is no exception text before the HTML marker, the converted
    body stands alone instead of gaining a leading blank line.
    """
    error = "<!DOCTYPE html><html><body><h1>502 Bad Gateway</h1></body></html>"
    rendered = str(_error_body(error))

    assert "502 Bad Gateway" in rendered
    assert "<!DOCTYPE" not in rendered


def test_error_body_returns_original_when_conversion_fails(monkeypatch):
    """A crashing HTML converter falls back to the raw error text.

    The panel must never hide the provider's message behind a converter
    bug, so _error_body returns the input untouched on failure.
    """
    import interpreter.core.respond as respond_module

    class _BoomConverter:
        body_width = 0

        def handle(self, *_args, **_kwargs):
            raise ValueError("converter blew up")

    monkeypatch.setattr(respond_module.html2text, "HTML2Text", _BoomConverter)
    error = "litellm.APIError: Problem - <!DOCTYPE html><html><body>x</body></html>"

    assert _error_body(error) == error
