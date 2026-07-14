import getpass
import platform

from interpreter.core.default_system_message import default_system_message


def test_default_system_message_is_nonempty_string():
    """default_system_message is a non-empty string instructing the model."""
    assert isinstance(default_system_message, str)
    assert default_system_message.strip()


def test_default_system_message_identifies_open_interpreter():
    """The system message declares the assistant as Open Interpreter."""
    assert "Open Interpreter" in default_system_message


def test_default_system_message_includes_user_context():
    """The system message embeds the user's name and OS at import time."""
    assert getpass.getuser() in default_system_message
    assert platform.system() in default_system_message
