"""Tests for the synchronous OpenInterpreter class.

<<<<<<< ours
<<<<<<< ours
These tests cover the core interpreter logic:
the sync chat loop, message handling, conversation history, and the _respond_and_store
chunk assembly logic.
=======
These tests cover the core interpreter logic that was not covered by PR #246:
the sync chat loop, message handling, conversation history, and the _respond_and_store
chunk assembly logic. Tests document current behavior only — no source changes.
>>>>>>> theirs
=======
These tests cover the core interpreter logic that was not covered by PR #246:
the sync chat loop, message handling, conversation history, and the _respond_and_store
chunk assembly logic. Tests document current behavior only — no source changes.
>>>>>>> theirs
"""

import json
import os
from unittest import mock

import pytest

from interpreter import OpenInterpreter


def test_chat_with_stream_true_returns_generator():
    """chat(stream=True) returns a generator that yields streaming chunks."""
    interpreter = OpenInterpreter()
    expected_gen = iter([{"role": "assistant", "type": "message", "content": "reply"}])
    with mock.patch.object(interpreter, "_streaming_chat", return_value=expected_gen) as mock_stream:
        result = interpreter.chat(message="hello", stream=True)
    mock_stream.assert_called_once_with(message="hello", display=True)
    assert result is expected_gen


def test_chat_non_blocking_starts_thread():
    """chat(blocking=False) spawns a thread and returns immediately."""
    interpreter = OpenInterpreter()
    with mock.patch("threading.Thread") as mock_thread:
        result = interpreter.chat(message="hello", blocking=False)
    mock_thread.assert_called_once()
    mock_thread.return_value.start.assert_called_once()
    assert result is None


def test_chat_stream_false_pulls_from_stream():
    """chat(stream=False) pulls all chunks from the stream and returns new messages."""
    interpreter = OpenInterpreter()
    interpreter.messages = [{"role": "user", "content": "existing"}]
    interpreter.last_messages_count = 1

    def fake_streaming_chat(message=None, display=True):
        """Yield one chunk and append it to messages."""
        chunk = {"role": "assistant", "type": "message", "content": "reply"}
        interpreter.messages.append(chunk)
        yield chunk

    with mock.patch.object(interpreter, "_streaming_chat", side_effect=fake_streaming_chat):
        result = interpreter.chat(message="hello", stream=False, display=False)

    assert interpreter.responding is False
    assert isinstance(result, list)
    assert {"role": "assistant", "type": "message", "content": "reply"} in result


def test_streaming_chat_dict_message_gets_role():
    """A dict message without a role field gets role='user' added."""
    interpreter = OpenInterpreter()
    with mock.patch.object(interpreter, "_respond_and_store", return_value=iter([])):
        list(interpreter._streaming_chat(message={"type": "message", "content": "hi"}, display=False))
    assert interpreter.messages == [
        {"role": "user", "type": "message", "content": "hi"}
    ]


def test_streaming_chat_empty_string_message():
    """An empty string message is treated as a valid one-off message."""
    interpreter = OpenInterpreter()
    with mock.patch.object(interpreter, "_respond_and_store", return_value=iter([])):
        list(interpreter._streaming_chat(message="", display=False))
    assert interpreter.messages == [
        {"role": "user", "type": "message", "content": ""}
    ]


def test_streaming_chat_no_message_raises():
    """_streaming_chat with no message and display=False raises an exception."""
    interpreter = OpenInterpreter()
    with pytest.raises(Exception, match="requires a display"):
        list(interpreter._streaming_chat(display=False))


def test_streaming_chat_display_true_redirects_to_terminal_interface():
    """When display=True, _streaming_chat delegates to terminal_interface."""
    interpreter = OpenInterpreter()
    with mock.patch(
        "interpreter.core.core.terminal_interface",
        return_value=iter([{"type": "message", "content": "via terminal"}]),
    ) as mock_ti:
        chunks = list(interpreter._streaming_chat(message="hello", display=True))
    mock_ti.assert_called_once_with(interpreter, "hello")
    assert chunks == [{"type": "message", "content": "via terminal"}]


def test_respond_and_store_yields_start_end_flags():
    """_respond_and_store wraps message chunks in start/end flag pairs."""
    interpreter = OpenInterpreter()

    def fake_respond(interpreter):
        """Yield a single assistant message chunk."""
        yield {"role": "assistant", "type": "message", "content": "hello"}

    with mock.patch("interpreter.core.core.respond", side_effect=fake_respond):
        chunks = list(interpreter._respond_and_store())

    assert chunks[0] == {"role": "assistant", "type": "message", "start": True}
    assert chunks[1] == {"role": "assistant", "type": "message", "content": "hello"}
    assert chunks[2] == {"role": "assistant", "type": "message", "end": True}


def test_respond_and_store_skips_empty_content():
    """Chunks with empty content are skipped and not stored."""
    interpreter = OpenInterpreter()

    def fake_respond(interpreter):
        """Yield an empty chunk followed by a real one."""
        yield {"role": "assistant", "type": "message", "content": ""}
        yield {"role": "assistant", "type": "message", "content": "real"}

    with mock.patch("interpreter.core.core.respond", side_effect=fake_respond):
        chunks = list(interpreter._respond_and_store())

    assert {"role": "assistant", "type": "message", "content": ""} not in chunks
    assert {"role": "assistant", "type": "message", "content": "real"} in chunks
    assert {"role": "assistant", "type": "message", "content": ""} not in interpreter.messages


def test_respond_and_store_active_line_none_ends_code():
    """An active_line chunk with None content ends the current code block."""
    interpreter = OpenInterpreter()
    interpreter.messages = [
        {"role": "assistant", "type": "code", "format": "python", "content": "x = 1"}
    ]

    def fake_respond(interpreter):
        """Yield an active_line=None chunk to end code execution."""
        yield {"role": "computer", "type": "console", "format": "active_line", "content": None}

    with mock.patch("interpreter.core.core.respond", side_effect=fake_respond):
        list(interpreter._respond_and_store())

    assert interpreter.messages[-1] == {
        "role": "computer",
        "type": "console",
        "format": "output",
        "content": "",
    }


def test_respond_and_store_confirmation_yields_end_flag():
    """A confirmation chunk yields an end flag for the previous message type."""
    interpreter = OpenInterpreter()

    def fake_respond(interpreter):
        """Yield a message then a confirmation chunk."""
        yield {"role": "assistant", "type": "message", "content": "msg"}
        yield {"role": "computer", "type": "confirmation", "content": {"format": "python", "content": "x=1"}}

    with mock.patch("interpreter.core.core.respond", side_effect=fake_respond):
        chunks = list(interpreter._respond_and_store())

    end_flags = [c for c in chunks if c.get("end") is True]
    assert len(end_flags) >= 1
    assert end_flags[0]["type"] == "message"


def test_respond_and_store_confirmation_auto_run_skips_yield():
    """When auto_run is True, confirmation chunks are not yielded to the caller."""
    interpreter = OpenInterpreter()
    interpreter.auto_run = True

    def fake_respond(interpreter):
        """Yield a confirmation chunk when auto_run is enabled."""
        yield {"role": "computer", "type": "confirmation", "content": {"format": "python", "content": "x=1"}}

    with mock.patch("interpreter.core.core.respond", side_effect=fake_respond):
        chunks = list(interpreter._respond_and_store())

    confirmation_chunks = [c for c in chunks if c.get("type") == "confirmation"]
    assert len(confirmation_chunks) == 0


def test_respond_and_store_truncates_console_output():
    """Console output chunks are truncated when they exceed max_output."""
    interpreter = OpenInterpreter()
    interpreter.max_output = 10
    interpreter.computer.import_computer_api = False

    def fake_respond(interpreter):
        """Yield a long console output chunk."""
        yield {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": "a" * 100,
        }

    with mock.patch("interpreter.core.core.respond", side_effect=fake_respond):
        list(interpreter._respond_and_store())

    assert "Output truncated" in interpreter.messages[-1]["content"]
    assert "a" * 100 not in interpreter.messages[-1]["content"]


def test_respond_and_store_ephemeral_chunks_not_stored():
    """Review and active_line chunks are not stored in messages."""
    interpreter = OpenInterpreter()

    def fake_respond(interpreter):
        """Yield ephemeral chunks that should not be stored."""
        yield {"role": "assistant", "type": "review", "content": "looks good"}
        yield {"role": "computer", "type": "console", "format": "active_line", "content": 1}

    with mock.patch("interpreter.core.core.respond", side_effect=fake_respond):
        list(interpreter._respond_and_store())

    stored_types = [m.get("type") for m in interpreter.messages]
    assert "review" not in stored_types
    stored_formats = [m.get("format") for m in interpreter.messages]
    assert "active_line" not in stored_formats


def test_will_contribute_property():
    """will_contribute is True only when all contributing conditions are met."""
    interpreter = OpenInterpreter()
    interpreter.contribute_conversation = True
    interpreter.offline = False
    interpreter.conversation_history = True
    interpreter.disable_telemetry = False
    assert interpreter.will_contribute is True

    interpreter.offline = True
    assert interpreter.will_contribute is False
    interpreter.offline = False
    interpreter.conversation_history = False
    assert interpreter.will_contribute is False
    interpreter.conversation_history = True
    interpreter.disable_telemetry = True
    assert interpreter.will_contribute is False


def test_display_message_plain_text_mode(capsys):
    """In plain_text_display mode, display_message prints directly."""
    interpreter = OpenInterpreter()
    interpreter.plain_text_display = True
    interpreter.display_message("**bold** text")
    assert "**bold** text" in capsys.readouterr().out


def test_display_message_markdown_mode():
    """When not in plain_text mode, display_message delegates to display_markdown_message."""
    interpreter = OpenInterpreter()
    interpreter.plain_text_display = False
    with mock.patch("interpreter.core.core.display_markdown_message") as mock_display:
        interpreter.display_message("**bold** text")
    mock_display.assert_called_once_with("**bold** text")


def test_get_oi_dir_returns_oi_dir():
    """get_oi_dir returns the oi_dir path for use in profiles."""
    from interpreter.terminal_interface.utils.oi_dir import oi_dir

    interpreter = OpenInterpreter()
    assert interpreter.get_oi_dir() == oi_dir


def test_conversation_history_saved(tmp_path, monkeypatch):
    """When conversation_history is on, chat saves messages to a JSON file."""
    interpreter = OpenInterpreter()
    interpreter.conversation_history = True
    interpreter.conversation_history_path = str(tmp_path)
    interpreter.conversation_filename = "test_conv.json"

    with mock.patch.object(interpreter, "_respond_and_store", return_value=iter([])):
        interpreter.chat(message="hello", display=False)

    saved_path = tmp_path / "test_conv.json"
    assert saved_path.exists()
    with open(saved_path) as f:
        saved = json.load(f)
    assert saved[-1]["content"] == "hello"


def test_conversation_history_generates_filename(tmp_path, monkeypatch):
    """First chat without a conversation_filename generates one from the first message."""
    interpreter = OpenInterpreter()
    interpreter.conversation_history = True
    interpreter.conversation_history_path = str(tmp_path)
    interpreter.conversation_filename = None

    with mock.patch.object(interpreter, "_respond_and_store", return_value=iter([])):
        interpreter.chat(message="hello world", display=False)

    assert interpreter.conversation_filename is not None
    assert interpreter.conversation_filename.endswith(".json")
    assert "hello" in interpreter.conversation_filename


def test_max_output_rejects_boolean():
    """max_output rejects booleans even though bool is a subclass of int."""
    with pytest.raises(ValueError, match="positive integer"):
        OpenInterpreter(max_output=True)


def test_max_output_setter_rejects_invalid():
    """max_output setter rejects invalid values after construction."""
    interpreter = OpenInterpreter()
    with pytest.raises(ValueError, match="positive integer"):
        interpreter.max_output = -5
    with pytest.raises(ValueError, match="positive integer"):
        interpreter.max_output = "large"


def test_reset_resets_last_messages_count():
    """reset() clears last_messages_count along with messages."""
    interpreter = OpenInterpreter()
    interpreter.messages = [{"role": "user", "content": "hi"}]
    interpreter.last_messages_count = 1
    with mock.patch.object(interpreter.computer, "terminate"):
        interpreter.reset()
    assert interpreter.last_messages_count == 0


def test_chat_handles_generator_exit():
    """chat() resets responding flag when GeneratorExit is raised."""
    interpreter = OpenInterpreter()

    def raise_generator_exit(*args, **kwargs):
        """Simulate the generator being closed early."""
        raise GeneratorExit()

    with mock.patch.object(interpreter, "_streaming_chat", side_effect=raise_generator_exit):
        try:
            interpreter.chat(message="hello", stream=True)
        except GeneratorExit:
            pass
    assert interpreter.responding is False


def test_chat_sends_telemetry_on_error():
    """When chat errors and telemetry is on, an error telemetry event is sent."""
    interpreter = OpenInterpreter()
    interpreter.disable_telemetry = False
    interpreter.offline = False

    def raise_error(*args, **kwargs):
        """Simulate an error during streaming."""
        raise ValueError("API failed")

    with mock.patch.object(interpreter, "_streaming_chat", side_effect=raise_error):
        with mock.patch("interpreter.core.core.send_telemetry") as mock_telemetry:
            try:
                interpreter.chat(message="hello", stream=True)
            except ValueError:
                pass
    error_calls = [
        c for c in mock_telemetry.call_args_list if c[0][0] == "errored"
    ]
    assert len(error_calls) == 1
    assert error_calls[0][1]["properties"]["error"] == "API failed"
