from unittest import mock

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
    interpreter.disable_telemetry = False
    interpreter.offline = False
    assert interpreter.anonymous_telemetry is True
    interpreter.offline = True
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
