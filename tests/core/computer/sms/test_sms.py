from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import interpreter.core.computer.sms.sms as sms_module
from interpreter.core.computer.sms.sms import SMS
from tests.helpers import patch_expanduser


def test_send_non_macos_prints_message(capsys):
    """SMS.send() prints a Mac-only notice and returns None on non-macOS platforms."""
    with mock.patch("sys.platform", "linux"):
        sms = SMS(computer=SimpleNamespace())
        assert sms.send("+1", "hi") is None
    assert "Only supported on Mac" in capsys.readouterr().out


def test_resolve_database_path(monkeypatch, tmp_path):
    """On macOS, chat.db lives under expanduser('~')/Library/Messages."""
    patch_expanduser(monkeypatch, sms_module, tmp_path)
    with mock.patch("sys.platform", "darwin"):
        sms = SMS(computer=SimpleNamespace())
    assert Path(sms.database_path) == (tmp_path / "Library" / "Messages" /
                                       "chat.db")


def test_get_non_macos(capsys):
    """SMS.get() prints a Mac-only notice and returns None on non-macOS platforms."""
    with mock.patch("sys.platform", "linux"):
        sms = SMS(computer=SimpleNamespace())
        assert sms.get() is None
    assert "Only supported on Mac" in capsys.readouterr().out


def _run_send(message):
    """Run SMS.send() on macOS with osascript stubbed; return (script, return)."""
    sms = SMS(computer=SimpleNamespace())
    with mock.patch("sys.platform", "darwin"), mock.patch("subprocess.run") as run:
        result = sms.send("+15551234", message)
    script = run.call_args[0][0][2]
    return script, result


def test_send_escapes_backslashes_before_quotes():
    """Backslashes are escaped before quotes, so a trailing backslash survives (issue #412).

    Doing it the other way double-escapes the backslash that was just
    introduced for a quote, and one immediately before a quote escapes the
    quote instead.
    """
    script, result = _run_send("back\\slash")

    assert "back\\\\slash" in script
    assert result == "Message sent successfully"


def test_send_escapes_quotes():
    """Double quotes are escaped so they do not terminate the AppleScript string."""
    script, _ = _run_send('say "hi"')

    assert 'say \\"hi\\"' in script


def test_send_escapes_backslash_immediately_before_quote():
    """A backslash just before a quote must not swallow that quote's escaping."""
    script, _ = _run_send('path\\"end')

    # The backslash becomes a literal escaped pair, and the quote is still
    # escaped for AppleScript — both, not one at the expense of the other.
    assert 'path\\\\\\"end' in script
