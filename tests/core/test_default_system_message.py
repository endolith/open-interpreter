import getpass
import os
import platform

from interpreter import OpenInterpreter
from interpreter.core.default_system_message import build_default_system_message


def test_default_system_message_is_nonempty_string():
    """The builder returns a non-empty string instructing the model."""
    message = build_default_system_message()
    assert isinstance(message, str)
    assert message.strip()


def test_default_system_message_identifies_open_interpreter():
    """The system message declares the assistant as Open Interpreter."""
    assert "Open Interpreter" in build_default_system_message()


def test_default_system_message_includes_user_context():
    """The system message embeds the user's name and OS."""
    message = build_default_system_message()
    assert getpass.getuser() in message
    assert platform.system() in message


def test_default_system_message_includes_cwd():
    """The model is told the directory it was started in, so relative paths resolve correctly."""
    assert os.getcwd() in build_default_system_message()


def test_interpreter_captures_cwd_at_initialization(monkeypatch, tmp_path):
    """Changing directory after import must be reflected in a new session's message."""
    monkeypatch.chdir(tmp_path)

    interpreter = OpenInterpreter(offline=True, disable_telemetry=True)

    assert str(tmp_path) in interpreter.system_message
