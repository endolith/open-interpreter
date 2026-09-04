"""Tests for start_terminal_interface.main() — CLI entry and contribution flow."""

from unittest import mock

import pytest

from interpreter.terminal_interface import start_terminal_interface as sti


def _patch_interpreter():
    """Create a mock interpreter and a patch context that injects it into main()."""
    interpreter = mock.MagicMock()
    interpreter.offline = False
    interpreter.disable_telemetry = False
    interpreter.messages = [1, 2, 3, 4, 5]
    interpreter.contribute_conversation = False
    interpreter.llm.model = "gpt-4o"
    return interpreter, mock.patch("interpreter.interpreter", interpreter)


def test_main_calls_start_terminal_interface():
    """main() passes the module interpreter to start_terminal_interface.

    main() must hand the configured interpreter instance to the REPL entry
    point so the session runs with the user's settings; a mismatch here would
    silently run startup with a different (default) interpreter.
    """
    interpreter, patch_interp = _patch_interpreter()
    with mock.patch.object(sti, "start_terminal_interface") as start:
        with patch_interp:
            sti.main()
    start.assert_called_once_with(interpreter)


def test_main_terminates_computer_on_success():
    """main() terminates the computer in the finally block after a normal exit.

    The computer session must be cleaned up when main() returns so no live
    terminal/display session is left running after the CLI exits.
    """
    interpreter, patch_interp = _patch_interpreter()
    with mock.patch.object(sti, "start_terminal_interface"):
        with patch_interp:
            sti.main()
    interpreter.computer.terminate.assert_called_once()


def test_main_handles_keyboard_interrupt_with_feedback():
    """main() prompts for feedback on KeyboardInterrupt when messages > 3.

    After an interrupted session with several messages, main() must still offer
    the feedback/contribution flow so the user can report on a completed
    conversation instead of losing that opportunity.
    """
    interpreter, patch_interp = _patch_interpreter()
    interpreter.offline = False
    interpreter.disable_telemetry = False

    with mock.patch.object(sti, "contribute_conversations") as contribute:
        with mock.patch.object(sti, "start_terminal_interface", side_effect=KeyboardInterrupt):
            with patch_interp:
                with mock.patch("builtins.input", return_value="y"):
                    sti.main()

    assert interpreter.contribute_conversation is True
    contribute.assert_called_once()


def test_main_handles_keyboard_interrupt_decline():
    """main() does not contribute when user says 'n'.

    Declining contribution must be respected so a user who opts out of sharing
    a conversation never has it uploaded behind their back.
    """
    interpreter, patch_interp = _patch_interpreter()

    with mock.patch.object(sti, "contribute_conversations") as contribute:
        with mock.patch.object(sti, "start_terminal_interface", side_effect=KeyboardInterrupt):
            with patch_interp:
                with mock.patch("builtins.input", return_value="n"):
                    sti.main()

    assert interpreter.contribute_conversation is False
    contribute.assert_not_called()


def test_main_nested_keyboard_interrupt_in_feedback():
    """main() survives a second KeyboardInterrupt inside the feedback prompt.

    If the user interrupts while being asked for feedback, main() must not
    crash the shutdown path — the CLI should still exit cleanly.
    """
    interpreter, patch_interp = _patch_interpreter()

    with mock.patch.object(sti, "start_terminal_interface", side_effect=KeyboardInterrupt):
        with patch_interp:
            with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
                sti.main()


def test_main_terminates_on_exception():
    """main() terminates the computer even when startup raises.

    Cleanup runs in the finally block, so a failed startup (e.g. a broken
    import) must not leave a live computer session behind.
    """
    interpreter, patch_interp = _patch_interpreter()

    with mock.patch.object(sti, "start_terminal_interface", side_effect=RuntimeError("boom")):
        with patch_interp:
            with pytest.raises(RuntimeError):
                sti.main()

    interpreter.computer.terminate.assert_called_once()


def test_main_no_feedback_when_few_messages():
    """main() skips the feedback prompt for short conversations.

    A session with 3 or fewer messages is too brief to warrant a feedback
    prompt, so main() must not interrupt the user's exit with one.
    """
    interpreter, patch_interp = _patch_interpreter()
    interpreter.messages = [1, 2, 3]

    with mock.patch.object(sti, "start_terminal_interface", side_effect=KeyboardInterrupt):
        with patch_interp:
            with mock.patch("builtins.input") as mock_input:
                sti.main()

    mock_input.assert_not_called()


def test_main_no_feedback_when_offline():
    """main() skips feedback when the interpreter is offline.

    Offline mode must not prompt for telemetry feedback since the user has
    opted out of telemetry entirely.
    """
    interpreter, patch_interp = _patch_interpreter()
    interpreter.offline = True

    with mock.patch.object(sti, "start_terminal_interface", side_effect=KeyboardInterrupt):
        with patch_interp:
            with mock.patch("builtins.input") as mock_input:
                sti.main()

    mock_input.assert_not_called()


def test_main_contribute_when_model_is_i():
    """After 'y' feedback, model 'i' uploads without a second permission prompt.

    main() always asks for feedback first; for model 'i' it then skips the
    second contribution-permission prompt (since that model implies consent)
    and uploads the conversation. This keeps the flow smooth for the built-in
    model while still recording the user's initial feedback choice.
    """
    interpreter, patch_interp = _patch_interpreter()
    interpreter.llm.model = "i"
    interpreter.contribute_conversation = False

    with mock.patch.object(sti, "contribute_conversations") as contribute:
        with mock.patch.object(sti, "start_terminal_interface", side_effect=KeyboardInterrupt):
            with patch_interp:
                with mock.patch("builtins.input", return_value="y") as mock_input:
                    sti.main()

    assert interpreter.contribute_conversation is True
    contribute.assert_called_once()
    mock_input.assert_called_once()
