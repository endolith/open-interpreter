"""Typing a word to leave the session.

Ctrl-C and Ctrl-D already exit, but nothing said so: %help listed every other
command and not that one. So a user who typed the obvious command got the worst
possible answer — `%exit` and `%quit` reached handle_magic_command's
unknown-command fallback, so the session looked like it had stopped responding
to input. Bare words and slash forms stay messages: either can be a legitimate
answer to the model (a command name, a filesystem path).
"""

import pytest

from interpreter.terminal_interface.terminal_interface import EXIT_WORDS, is_exit_command


@pytest.mark.parametrize(
    "typed",
    ["%exit", "%quit", "  %exit  ", "%EXIT", "%Quit", "\t%quit\n"],
)
def test_an_exit_word_ends_the_session(typed):
    """The magic-command prefix, in any case, with surrounding whitespace, is recognised.

    `%` is the existing magic-command precedent, so it is the only prefix that
    exits.
    """
    assert is_exit_command(typed, interactive=True)


@pytest.mark.parametrize(
    "typed", ["exit", "quit", "Exit", "QUIT", " exit ", "/exit", "/quit", "/EXIT"]
)
def test_a_bare_or_slash_word_is_still_a_message(typed):
    """Bare words and slash forms are left for the model, not taken as commands.

    The model may ask a direct question whose answer is exactly that word
    ("which command closes vim?" -> "exit"), and a slash form can be a
    filesystem path in an answer. Treating either as a command would kill an
    ongoing project by accident; only the magic-command prefix exits.
    """
    assert not is_exit_command(typed, interactive=True)


@pytest.mark.parametrize(
    "typed",
    ["exit the loop early", "how do I quit vim?", "/exits", "quitting", "exit()", ""],
)
def test_a_message_that_merely_mentions_exiting_is_still_sent(typed):
    """Only the bare line exits; a sentence containing it is a normal message."""
    assert not is_exit_command(typed, interactive=True)


@pytest.mark.parametrize("typed", sorted(EXIT_WORDS))
def test_exit_words_are_ignored_when_not_interactive(typed):
    """A piped or --stdin message saying "quit" is data, not a command.

    Non-interactive callers have no prompt to return to, so treating their
    input as a control word would end the run on ordinary content.
    """
    assert not is_exit_command(typed, interactive=False)


def test_non_string_input_is_not_an_exit_command():
    """The message may be a list of LMC dicts from the Python API, not a string."""
    assert not is_exit_command([{"role": "user", "content": "quit"}], interactive=True)


def test_the_guard_runs_before_the_message_reaches_the_model():
    """The exit check sits ahead of the magic-command and chat dispatch.

    If it ran after, an exit word would be handled as a magic command or sent
    to the model first, which is the bug being fixed.
    """
    import inspect

    from interpreter.terminal_interface import terminal_interface as module

    source = inspect.getsource(module.terminal_interface)
    assert source.index("is_exit_command(") < source.index('message.startswith("%")')
    assert source.index("is_exit_command(") < source.index("interpreter.chat(")


def _fake_interpreter():
    """Build the minimal interpreter stand-in the interactive loop touches."""
    from unittest import mock

    fake = mock.Mock()
    fake.messages = []
    fake.auto_run = True  # skip the intro message so only loop messages are recorded
    fake.multi_line = False
    fake.display_message = mock.Mock()
    fake.verbose = False
    fake.os = False
    # The loop iterates chat's chunks; an empty stream lets it return to input.
    fake.chat.side_effect = lambda *args, **kwargs: iter(())
    return fake


def test_terminal_interface_exits_when_a_typed_line_is_an_exit_word(monkeypatch):
    """The interactive loop turns a typed %exit into KeyboardInterrupt.

    The predicate tests above cover the decision; this covers the wiring that
    actually ends the session and the "Exiting..." message the user sees, which
    is the half of the feature a unit test of the predicate cannot reach.
    """
    import interpreter.terminal_interface.terminal_interface as ti

    fake = _fake_interpreter()
    monkeypatch.setattr("builtins.input", lambda prompt="": "%exit")

    # terminal_interface is a generator (it yields chat chunks), so driving it
    # one step runs the loop up to the first yield. The exit branch raises
    # before that yield, which is the behavior under test.
    generator = ti.terminal_interface(fake, None)
    with pytest.raises(KeyboardInterrupt):
        next(generator)

    fake.display_message.assert_called_once_with("\n\n`Exiting...`")


def test_terminal_interface_sends_a_non_exit_word_to_chat(monkeypatch):
    """A non-exit word falls through the exit check to the model.

    The loop's exit predicate has two branches: an exit word ends the session,
    and any other word continues it. This drives the continuing branch, which
    an exit-only test leaves untested, then confirms the next word still exits.
    """
    import interpreter.terminal_interface.terminal_interface as ti

    fake = _fake_interpreter()
    answers = iter(["hello", "%exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    with pytest.raises(KeyboardInterrupt):
        next(ti.terminal_interface(fake, None))

    assert fake.chat.call_args.args[0] == "hello"
    fake.display_message.assert_called_once_with("\n\n`Exiting...`")
