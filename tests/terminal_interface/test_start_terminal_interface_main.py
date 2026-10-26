"""Tests for start_terminal_interface.main() — CLI entry and shutdown."""

from unittest import mock

import pytest

from interpreter.terminal_interface import start_terminal_interface as sti


def _patch_interpreter():
    """Create a mock interpreter and a patch context that injects it into main()."""
    interpreter = mock.MagicMock()
    interpreter.offline = False
    interpreter.disable_telemetry = False
    interpreter.messages = [1, 2, 3, 4, 5]
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
    interpreter.terminal.terminate.assert_called_once()


def test_main_terminates_computer_on_keyboard_interrupt():
    """main() terminates the computer when the session is Ctrl-C'd.

    An interrupted session must still clean up its computer session, or the
    CLI leaves a live terminal/display session behind on exit.
    """
    interpreter, patch_interp = _patch_interpreter()

    with mock.patch.object(
        sti, "start_terminal_interface", side_effect=KeyboardInterrupt
    ):
        with patch_interp:
            sti.main()

    interpreter.terminal.terminate.assert_called_once()


def test_main_does_not_prompt_after_keyboard_interrupt():
    """main() never prompts the user on the way out.

    The conversation-contribution service was shut down, so the old exit flow
    (feedback question, then a "send us this chat" upsell) had nothing left to
    upload. It is removed, so an interrupted session must exit without blocking
    on any question — a prompt here would hang scripted runs that Ctrl-C out.
    """
    interpreter, patch_interp = _patch_interpreter()

    with mock.patch.object(
        sti, "start_terminal_interface", side_effect=KeyboardInterrupt
    ):
        with patch_interp:
            with mock.patch("builtins.input") as mock_input:
                sti.main()

    mock_input.assert_not_called()


def test_main_terminates_on_exception():
    """main() terminates the computer even when startup raises.

    Cleanup runs in the finally block, so a failed startup (e.g. a broken
    import) must not leave a live computer session behind.
    """
    interpreter, patch_interp = _patch_interpreter()

    with mock.patch.object(
        sti, "start_terminal_interface", side_effect=RuntimeError("boom")
    ):
        with patch_interp:
            with pytest.raises(SystemExit):
                sti.main()

    interpreter.terminal.terminate.assert_called_once()