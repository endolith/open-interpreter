from types import SimpleNamespace
from unittest import mock

from interpreter.core.render_message import render_message


def test_message_without_templates_returned_unchanged():
    """Messages with no {{...}} template blocks are returned verbatim without running code."""
    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=mock.Mock()),
        verbose=False,
        debug=False,
    )
    assert render_message(interpreter, "Plain system message") == "Plain system message"


def test_template_replaced_with_code_output():
    """{{...}} template blocks are executed and replaced with the console output from computer.run."""
    def fake_run(language, code, display=False):
        yield {"format": "output", "content": "hi"}

    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=fake_run),
        verbose=False,
        debug=False,
    )
    result = render_message(interpreter, 'Prefix {{print("hi")}} suffix')
    assert result == "Prefix hi suffix"


def test_save_skills_disabled_during_template_execution():
    """Template blocks run with save_skills=False so skill side effects are skipped."""
    save_skills_during_run = []

    def fake_run(language, code, display=False):
        save_skills_during_run.append(computer.save_skills)
        yield {"format": "output", "content": "42"}

    computer = SimpleNamespace(save_skills=True, run=fake_run)
    interpreter = SimpleNamespace(computer=computer, verbose=False, debug=False)
    result = render_message(interpreter, "Answer: {{1+1}}")
    assert result == "Answer: 42"
    assert save_skills_during_run == [False]
    assert computer.save_skills is True


def test_save_skills_restored_after_render_without_templates():
    """Rendering a message without templates leaves computer.save_skills unchanged."""
    computer = SimpleNamespace(save_skills=True, run=lambda *a, **k: iter([]))
    interpreter = SimpleNamespace(computer=computer, verbose=False, debug=False)
    render_message(interpreter, "no templates")
    assert computer.save_skills is True
