import itertools
from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.respond import respond


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
        llm=SimpleNamespace(run=lambda msgs: iter([]), supports_vision=False),
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
        max_budget=0,
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


def test_respond_handles_budget_exceeded():
    """respond() reports the session/max budget and stops on BudgetExceededError."""
    import litellm

    interpreter = _message_interpreter()
    interpreter.max_budget = 5

    def run(msgs):
        raise litellm.exceptions.BudgetExceededError(current_cost=0, max_budget=5)

    interpreter.llm.run = run
    list(respond(interpreter))

    assert "Max budget exceeded" in interpreter.display_message.call_args[0][0]


def test_respond_auth_error_raises_with_key_instructions():
    """Auth/api-key errors are re-raised with instructions for resetting the key."""
    interpreter = _message_interpreter()
    interpreter.offline = False

    def run(msgs):
        raise Exception("invalid api key")

    interpreter.llm.run = run
    with pytest.raises(Exception, match="To reset your API key"):
        list(respond(interpreter))


def test_respond_rate_limit_quota_shows_billing_message():
    """Rate-limit/quota errors print the OpenAI billing hint instead of raising."""
    import litellm

    interpreter = _message_interpreter()

    def run(msgs):
        raise litellm.exceptions.RateLimitError(
            "You exceeded your current quota", "openai", "gpt-4o"
        )

    interpreter.llm.run = run
    with mock.patch("interpreter.core.respond.display_markdown_message") as dm:
        list(respond(interpreter))

    dm.assert_called_once()
    assert "quota" in dm.call_args[0][0].lower()


def test_respond_model_access_prompt_switches_to_i_model():
    """Accepting the 'no access' prompt switches the model to hosted `i`."""
    interpreter = _message_interpreter()
    interpreter.offline = False
    interpreter.llm.model = "gpt-4o"

    def run(msgs):
        raise Exception("You do not have access to this model")

    interpreter.llm.run = run
    with mock.patch("builtins.input", return_value="y"):
        list(respond(interpreter))

    assert interpreter.llm.model == "i"
    assert "Model set to `i`" in interpreter.display_message.call_args_list[0][0][0]


def test_respond_model_access_prompt_no_raises():
    """Declining the 'no access' prompt re-raises the original error."""
    interpreter = _message_interpreter()
    interpreter.offline = False
    interpreter.llm.model = "gpt-4o"

    def run(msgs):
        raise Exception("You do not have access to this model")

    interpreter.llm.run = run
    with mock.patch("builtins.input", return_value="n"):
        with pytest.raises(Exception, match="have access"):
            list(respond(interpreter))


def test_respond_offline_errors_are_re_raised():
    """Offline sessions re-raise model-access errors rather than prompting."""
    interpreter = _message_interpreter()
    interpreter.llm.model = "gpt-4o"

    def run(msgs):
        raise Exception("You do not have access to this model")

    interpreter.llm.run = run
    with mock.patch("builtins.input") as prompt:
        with pytest.raises(Exception, match="have access"):
            list(respond(interpreter))

    prompt.assert_not_called()


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
