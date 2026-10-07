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


def test_max_output_must_be_positive_integer():
    """max_output rejects zero, negatives, and non-integers at construction and assignment."""
    with pytest.raises(ValueError, match="positive integer"):
        OpenInterpreter(max_output=0)
    with pytest.raises(ValueError, match="positive integer"):
        OpenInterpreter(max_output=-100)

    interpreter = OpenInterpreter()
    with pytest.raises(ValueError, match="positive integer"):
        interpreter.max_output = 0


def test_skills_path_overrides_skills_directory(tmp_path):
    """skills_path= points the computer's skills at the given directory."""
    interpreter = OpenInterpreter(skills_path=str(tmp_path))
    assert interpreter.computer.skills.path == str(tmp_path)


def test_local_setup_delegates_to_wizard():
    """local_setup() hands the interpreter to the terminal wizard function."""
    interpreter = OpenInterpreter()
    with mock.patch(
        "interpreter.core.core.local_setup", return_value=interpreter
    ) as wizard:
        assert interpreter.local_setup() is None
    wizard.assert_called_once_with(interpreter)


def test_wait_returns_messages_since_last_count():
    """wait() returns only the messages appended after last_messages_count."""
    interpreter = OpenInterpreter()
    interpreter.responding = False
    interpreter.messages = [{"a": 1}, {"b": 2}]
    interpreter.last_messages_count = 1
    assert interpreter.wait() == [{"b": 2}]


def test_chat_creates_history_directory_and_saves(tmp_path):
    """The history branch creates a missing directory and writes the JSON file."""
    import json

    interpreter = OpenInterpreter(
        disable_telemetry=True,
        conversation_history=True,
        conversation_history_path=str(tmp_path / "new-dir"),
    )
    interpreter.messages = [{"role": "user", "content": "hello world today"}]
    with mock.patch.object(
        interpreter, "_respond_and_store", return_value=iter([])
    ):
        interpreter.chat("hello world today", display=False)
    saved = list((tmp_path / "new-dir").glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text()) == interpreter.messages


def test_respond_and_store_stops_on_stop_event(capsys):
    """A set stop_event breaks the respond loop before the first chunk is stored."""
    interpreter = OpenInterpreter()
    interpreter.stop_event = mock.Mock()
    interpreter.stop_event.is_set.return_value = True
    chunks = iter([{"type": "message", "content": "never stored"}])
    with mock.patch("interpreter.core.core.respond", return_value=chunks):
        list(interpreter._respond_and_store())
    assert "Open Interpreter stopping." in capsys.readouterr().out
    assert interpreter.messages == []
