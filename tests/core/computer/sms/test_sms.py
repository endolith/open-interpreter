from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import interpreter.core.computer.sms.sms as sms_module
from interpreter.core.computer.sms.sms import SMS

from tests.helpers import patch_expanduser


def test_send_non_macos_prints_message(capsys):
    with mock.patch("sys.platform", "linux"):
        sms = SMS(computer=SimpleNamespace())
        assert sms.send("+1", "hi") is None
    assert "Only supported on Mac" in capsys.readouterr().out


def test_resolve_database_path(monkeypatch, tmp_path):
    """On macOS, chat.db lives under expanduser('~')/Library/Messages."""
    patch_expanduser(monkeypatch, sms_module, tmp_path)
    with mock.patch("sys.platform", "darwin"):
        sms = SMS(computer=SimpleNamespace())
    assert Path(sms.database_path) == tmp_path / "Library" / "Messages" / "chat.db"


def test_get_non_macos(capsys):
    with mock.patch("sys.platform", "linux"):
        sms = SMS(computer=SimpleNamespace())
        assert sms.get() is None
    assert "Only supported on Mac" in capsys.readouterr().out
