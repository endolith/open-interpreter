from interpreter.terminal_interface.contributing_conversations import (
    user_wants_to_contribute_future,
    user_wants_to_contribute_past,
)


def _answers(monkeypatch, *typed):
    """Feed `typed` to successive prompts, recording the prompt shown each time."""
    pending = iter(typed)
    shown = []

    def fake_input(prompt=""):
        shown.append(prompt)
        return next(pending)

    monkeypatch.setattr("builtins.input", fake_input)
    return shown


def test_contribute_past_reprompts_instead_of_reading_a_word(monkeypatch):
    """A near-miss like "Yay" must not silently answer "n" and skip the upload.

    The prompt guards uploading every past conversation for training, so a
    near-miss has to re-prompt rather than fall through to the "anything that
    isn't exactly y" branch, where a typo and a real "no" are indistinguishable.
    """
    shown = _answers(monkeypatch, "Yay", "n")
    assert user_wants_to_contribute_past() is False
    # Full prompt once, then the minimal reprompt.
    assert shown == ["(y/n) ", "  "]


def test_contribute_future_accepts_a_bare_y(monkeypatch):
    """A single "y" opts in, which is the answer the prompt asks for."""
    _answers(monkeypatch, "y")
    assert user_wants_to_contribute_future() is True


def test_contribute_future_accepts_a_bare_n(monkeypatch):
    """A single "n" opts out, as does a deliberate "no" only if retyped as n.

    Only exact choices resolve, so "no" is rejected like any other word.
    """
    _answers(monkeypatch, "no", "n")
    assert user_wants_to_contribute_future() is False
