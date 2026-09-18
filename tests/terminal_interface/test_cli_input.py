from interpreter.core.utils.prompt_choice import prompt_choice
from interpreter.terminal_interface.utils.cli_input import cli_input


def test_cli_input_strips_mouse_reports(monkeypatch):
    """Mouse-report bytes arriving mid-prompt never reach the caller."""
    monkeypatch.setattr(
        "builtins.input", lambda _prompt="": "hi\x1b[<35;25;57M\x1b[61;4R"
    )
    assert cli_input("> ") == "hi"


def test_cli_input_mouse_only_becomes_empty(monkeypatch):
    """A prompt holding only leaked reports reads as blank (ignored downstream)."""
    monkeypatch.setattr(
        "builtins.input", lambda _prompt="": "\x1b[<35;25;57M\x1b[61;4R"
    )
    assert cli_input("> ") == ""


def test_prompt_choice_ignores_leaked_report_bytes(monkeypatch):
    """Leaked tracking bytes around a y/n answer don't fail validation."""
    monkeypatch.setattr(
        "builtins.input", lambda _prompt="": "\x1b[<0;80;53mY\x1b[61;4R"
    )
    assert prompt_choice("  ", ("y", "n")) == "y"
