import io

import pytest

from interpreter.core.utils.prompt_choice import (
    NoInteractiveInput,
    prompt_choice,
    stdin_is_interactive,
)


class FakeStdin(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


@pytest.fixture
def tty(monkeypatch):
    monkeypatch.setattr("sys.stdin", FakeStdin(tty=True))


@pytest.fixture
def no_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin", FakeStdin(tty=False))


def test_returns_the_chosen_option_on_a_tty(tty, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert prompt_choice("run? ", ("y", "n")) == "y"


def test_reprompts_until_the_answer_is_one_of_the_choices(tty, monkeypatch):
    answers = iter(["", "q", "maybe", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert prompt_choice("run? ", ("y", "n")) == "n"


def test_raises_when_there_is_no_tty(no_tty):
    with pytest.raises(NoInteractiveInput):
        prompt_choice("run? ", ("y", "n"))


def test_raises_when_stdin_closes_mid_session(tty, monkeypatch):
    def closed(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)
    with pytest.raises(NoInteractiveInput):
        prompt_choice("run? ", ("y", "n"))


@pytest.mark.parametrize(
    "choices",
    [
        ("y", "n"),
        ("y", "a", "n"),
        ("f", "r", "n"),
        # The run prompt. Its last option is "edit" / "add to allowlist", so any
        # implementation that fell back to a choice here would open an editor or
        # permanently approve the command instead of declining.
        ("y", "n", "e"),
        ("y", "n", "e", "a"),
    ],
)
def test_never_answers_for_the_user(no_tty, choices):
    """
    With no TTY, prompt_choice must not return any choice at all.

    Returning one means some call site acts on an answer nobody gave, and which
    option that is depends on the order of `choices`.
    """
    with pytest.raises(NoInteractiveInput):
        prompt_choice("  ", choices)


def test_error_names_the_question_that_could_not_be_answered(no_tty):
    with pytest.raises(NoInteractiveInput) as excinfo:
        prompt_choice("  Would you like to run this code?\n\n  ", ("y", "n", "e"))
    message = str(excinfo.value)
    assert "Would you like to run this code?" in message
    assert "y/n/e" in message


def test_stdin_is_interactive_is_false_without_a_tty(no_tty):
    assert stdin_is_interactive() is False


def test_stdin_is_interactive_survives_a_detached_stdin(monkeypatch):
    class Detached:
        def isatty(self):
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr("sys.stdin", Detached())
    assert stdin_is_interactive() is False

    monkeypatch.setattr("sys.stdin", None)
    assert stdin_is_interactive() is False
