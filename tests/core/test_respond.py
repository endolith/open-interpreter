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
    )


def test_unsupported_language_yields_console_output():
    interpreter = _code_interpreter(language="brainfuck", code="++")
    chunks = list(itertools.islice(respond(interpreter), 3))
    outputs = [c for c in chunks if c.get("format") == "output"]
    assert outputs
    assert "disabled or not supported" in outputs[0]["content"]


def test_text_language_converted_to_assistant_message():
    interpreter = _code_interpreter(language="text", code="notes here")
    list(itertools.islice(respond(interpreter), 1))
    assert interpreter.messages[-1]["type"] == "message"
    assert "notes here" in interpreter.messages[-1]["content"]


def test_functions_execute_hallucination_parsed():
    code = 'functions.execute({"language": "python", "code": "7*6"})'
    interpreter = _code_interpreter(code=code)
    gen = respond(interpreter)
    list(itertools.islice(gen, 3))
    assert interpreter.messages[0]["content"] == "7*6"
    assert interpreter.messages[0]["format"] == "python"


def test_executeexecute_suffix_stripped():
    interpreter = _code_interpreter(code="print(1)executeexecute")
    list(itertools.islice(respond(interpreter), 3))
    assert interpreter.messages[0]["content"] == "print(1)"


def test_json_code_block_parsed():
    interpreter = _code_interpreter(code='{"language": "python", "code": "8*8"}')
    list(itertools.islice(respond(interpreter), 3))
    assert interpreter.messages[0]["content"] == "8*8"


def test_llm_run_when_last_message_not_code():
    interpreter = _code_interpreter()
    interpreter.messages = [{"role": "user", "type": "message", "content": "hi"}]
    interpreter.llm.run = lambda msgs: iter([{"type": "message", "content": "hello"}])
    chunks = list(respond(interpreter))
    assert {"role": "assistant", "type": "message", "content": "hello"} in chunks


def test_respond_requires_messages():
    interpreter = _code_interpreter()
    interpreter.messages = []
    with pytest.raises(AssertionError, match="User message was not passed"):
        next(respond(interpreter))
