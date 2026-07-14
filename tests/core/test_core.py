from unittest import mock

import pytest

from interpreter import OpenInterpreter


def test_reset_clears_messages():
    """reset() terminates the computer session and clears the conversation history."""
    interpreter = OpenInterpreter()
    interpreter.messages = [{"role": "user", "content": "hi"}]
    with mock.patch.object(interpreter.computer, "terminate") as terminate:
        interpreter.reset()
        terminate.assert_called_once()
    assert interpreter.messages == []


def test_anonymous_telemetry_property():
    """anonymous_telemetry is True only when telemetry is enabled and the interpreter is online."""
    interpreter = OpenInterpreter()
    assert interpreter.disable_telemetry is True
    assert interpreter.anonymous_telemetry is False
    interpreter.disable_telemetry = False
    interpreter.offline = False
    assert interpreter.anonymous_telemetry is True
    interpreter.offline = True
    assert interpreter.anonymous_telemetry is False


def test_chat_sends_telemetry_when_enabled():
    """chat() sends started_chat telemetry only when anonymous_telemetry is enabled."""
    interpreter = OpenInterpreter(disable_telemetry=False, offline=False)
    with mock.patch(
        "interpreter.core.core.send_telemetry"
    ) as send_telemetry, mock.patch.object(
        interpreter, "_streaming_chat", return_value=iter([])
    ):
        interpreter.chat(message="hi", display=False, stream=False, blocking=True)
    send_telemetry.assert_called_once()
    assert send_telemetry.call_args.args[0] == "started_chat"


def test_chat_skips_telemetry_when_disabled_by_default():
    """chat() does not send telemetry when disable_telemetry is left at the default."""
    interpreter = OpenInterpreter()
    with mock.patch(
        "interpreter.core.core.send_telemetry"
    ) as send_telemetry, mock.patch.object(
        interpreter, "_streaming_chat", return_value=iter([])
    ):
        interpreter.chat(message="hi", display=False, stream=False, blocking=True)
    send_telemetry.assert_not_called()


def test_offline_mode_disables_telemetry_even_when_enabled():
    """Offline mode disables telemetry even when disable_telemetry is False."""
    interpreter = OpenInterpreter(disable_telemetry=False, offline=True)
    assert interpreter.anonymous_telemetry is False


def test_streaming_chat_string_message_appended():
    """_streaming_chat with a string appends a user message before invoking _respond_and_store."""
    interpreter = OpenInterpreter()
    with mock.patch.object(interpreter, "_respond_and_store", return_value=iter([])):
        list(interpreter._streaming_chat(message="Hello", display=False))
    assert interpreter.messages == [
        {"role": "user", "type": "message", "content": "Hello"}
    ]


def test_streaming_chat_list_replaces_messages():
    """_streaming_chat with a message list replaces the entire conversation before responding."""
    interpreter = OpenInterpreter()
    new_messages = [{"role": "user", "type": "message", "content": "replaced"}]
    with mock.patch.object(interpreter, "_respond_and_store", return_value=iter([])):
        list(interpreter._streaming_chat(message=new_messages, display=False))
    assert interpreter.messages == new_messages


def test_max_output_must_be_positive_integer():
    """max_output rejects zero, negatives, and non-integers at construction and assignment."""
    with pytest.raises(ValueError, match="positive integer"):
        OpenInterpreter(max_output=0)
    with pytest.raises(ValueError, match="positive integer"):
        OpenInterpreter(max_output=-100)

    interpreter = OpenInterpreter()
    with pytest.raises(ValueError, match="positive integer"):
        interpreter.max_output = 0
