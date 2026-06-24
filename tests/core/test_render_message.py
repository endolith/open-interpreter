from types import SimpleNamespace
from unittest import mock

from interpreter.core.render_message import render_message


def test_message_without_templates_returned_unchanged():
    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=mock.Mock()),
        verbose=False,
        debug=False,
    )
    assert render_message(interpreter, "Plain system message") == "Plain system message"


def test_template_replaced_with_code_output():
    def fake_run(language, code, display=False):
        yield {"format": "output", "content": "hi"}

    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=fake_run),
        verbose=False,
        debug=False,
    )
    result = render_message(interpreter, 'Prefix {{print("hi")}} suffix')
    assert result == "Prefix hi suffix"


def test_save_skills_restored_after_render():
    computer = SimpleNamespace(save_skills=True, run=lambda *a, **k: iter([]))
    interpreter = SimpleNamespace(computer=computer, verbose=False, debug=False)
    render_message(interpreter, "no templates")
    assert computer.save_skills is True
