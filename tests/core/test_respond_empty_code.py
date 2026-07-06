import itertools
from types import SimpleNamespace
from unittest import mock

from interpreter.core.respond import respond


class FakeLanguage:
    name = "Python"
    file_extension = "py"


def _interpreter_with_code(code, language="python"):
    terminal = SimpleNamespace(
        languages=[FakeLanguage()],
        get_language=lambda lang: FakeLanguage() if lang == "python" else None,
    )
    return SimpleNamespace(
        system_message="",
        custom_instructions="",
        messages=[
            {"role": "assistant", "type": "code", "format": language, "content": code}
        ],
        computer=SimpleNamespace(
            terminal=terminal,
            import_computer_api=False,
            system_message="",
            run=lambda *a, **k: iter([]),
            verbose=False,
            debug=False,
            emit_images=False,
            max_output=2800,
            save_skills=True,
            to_dict=lambda: {},
        ),
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


def test_empty_code_block_does_not_spin_forever():
    interpreter = _interpreter_with_code("   ")
    chunks = list(itertools.islice(respond(interpreter), 5))
    assert len(chunks) == 1
    assert "empty" in chunks[0]["content"].lower()
