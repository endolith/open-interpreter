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


def test_prompt_choice_rejects_spelled_out_answer(monkeypatch):
    """"yes" is not an answer: it re-prompts instead of counting as "y".

    Truncating to the first letter would let a multi-key or pasted answer act
    as a deliberate choice, so the prompt demands a single keypress.
    """
    answers = iter(["yes", "y"])
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)
    assert prompt_choice("  Retry? (y/n) ", ("y", "n")) == "y"
    # The full prompt is shown once, then only the minimal reprompt.
    assert prompts == ["  Retry? (y/n) ", "  "]


def test_prompt_choice_rejects_multi_letter_run(monkeypatch):
    """A run of valid letters ("yqn") is not a choice either.

    Every letter is one the user could have meant, so reading the first one
    would guess on their behalf; the answer has to be unambiguous.
    """
    answers = iter(["yqn", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert prompt_choice("  ", ("y", "a", "n")) == "n"


def test_prompt_choice_rejects_blank_line(monkeypatch):
    """An empty line re-prompts rather than resolving to a choice."""
    answers = iter(["", "  ", "a"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert prompt_choice("  ", ("y", "a", "n")) == "a"


def test_prompt_choice_tolerates_case_and_padding(monkeypatch):
    """A single letter still answers regardless of case or stray spaces."""
    monkeypatch.setattr("builtins.input", lambda _prompt="": "  Y  ")
    assert prompt_choice("  ", ("y", "n")) == "y"
