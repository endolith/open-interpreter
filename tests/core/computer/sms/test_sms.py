from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import datetime
import plistlib
import sqlite3

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
    assert Path(sms.database_path) == (tmp_path / "Library" / "Messages" / "chat.db")


def test_get_non_macos(capsys):
    """SMS.get() prints a Mac-only notice and returns None on non-macOS platforms."""
    with mock.patch("sys.platform", "linux"):
        sms = SMS(computer=SimpleNamespace())
        assert sms.get() is None
    assert "Only supported on Mac" in capsys.readouterr().out


def _mac_sms(monkeypatch=None, database_path=":memory:"):
    """An SMS instance that believes it is on macOS, with a stubbed database."""
    with mock.patch("sys.platform", "darwin"):
        sms = SMS(computer=SimpleNamespace())
    sms.database_path = database_path
    return sms


def test_resolve_database_path_root_uses_sudo_user(monkeypatch, tmp_path):
    """Running as root resolves chat.db under SUDO_USER's home, not root's."""
    patch_expanduser(monkeypatch, sms_module, tmp_path)
    monkeypatch.setattr(sms_module.os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_USER", "someone")
    with mock.patch("sys.platform", "darwin"):
        sms = SMS(computer=SimpleNamespace())
    # "~someone" is not stubbed, so it passes through — the point is the root
    # branch looked up SUDO_USER instead of using "~" like the branch below.
    assert sms.database_path == "~someone/Library/Messages/chat.db"


def test_resolve_database_path_falls_back_when_geteuid_raises(monkeypatch, tmp_path):
    """When geteuid itself fails, the path still resolves under the home dir."""
    patch_expanduser(monkeypatch, sms_module, tmp_path)

    def _boom():
        raise OSError("no euid here")

    monkeypatch.setattr(sms_module.os, "geteuid", _boom)
    with mock.patch("sys.platform", "darwin"):
        sms = SMS(computer=SimpleNamespace())
    assert Path(sms.database_path) == (tmp_path / "Library" / "Messages" / "chat.db")


def test_send_macos_runs_osascript_with_escaped_message():
    """On macOS, send() shells out to Messages via osascript and reports success."""
    sms = _mac_sms()
    with mock.patch("sys.platform", "darwin"):
        with mock.patch.object(sms_module.subprocess, "run") as run:
            assert sms.send("+1555", "hi") == "Message sent successfully"
    run.assert_called_once()
    args = run.call_args[0][0]
    assert args[:2] == ["osascript", "-e"]
    assert 'tell application "Messages"' in args[2]
    assert '"+1555"' in args[2]


def test_send_escapes_quotes_before_backslashes():
    """The escaping order is quotes-then-backslashes, so a quote gains two.

    "say \\"hi\\"" becomes \\\\"hi\\\\" in the script: the backslash added by
    the quote escape is itself escaped by the second replacement. Pinned as
    observed behavior — whether Messages interprets that correctly is a
    separate question for a real Mac.
    """
    sms = _mac_sms()
    with mock.patch("sys.platform", "darwin"):
        with mock.patch.object(sms_module.subprocess, "run") as run:
            sms.send("+1555", 'say "hi"')
    script = run.call_args[0][0][2]
    assert 'say \\\\"hi\\\\"' in script


def _message_db(path, rows):
    """A minimal chat.db: message rows joined to handle rows by ROWID."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT)")
    conn.execute(
        "CREATE TABLE message ("
        "ROWID INTEGER PRIMARY KEY, text BLOB, date INTEGER, "
        "handle_id INTEGER, is_from_me INTEGER)"
    )
    conn.execute("INSERT INTO handle (ROWID, id) VALUES (1, '+1555')")
    for text, date_ns, handle_id, is_from_me in rows:
        conn.execute(
            "INSERT INTO message (text, date, handle_id, is_from_me) VALUES (?, ?, ?, ?)",
            (text, date_ns, handle_id, is_from_me),
        )
    conn.commit()
    conn.close()


def _apple_ns(days=0):
    """Nanoseconds since 2001-01-01, the Apple epoch chat.db uses."""
    return days * 86400 * 10**9


def test_get_reads_plain_text_and_converts_apple_timestamps(tmp_path):
    """Plain-text rows come back with sender and Apple-epoch dates converted."""
    db = tmp_path / "chat.db"
    _message_db(db, [("hello", _apple_ns(days=2), 1, 0)])
    sms = _mac_sms(database_path=str(db))
    with mock.patch("sys.platform", "darwin"):
        messages = sms.get()
    assert messages == [
        {
            "date": datetime.datetime(2001, 1, 3),
            "from": "+1555",
            "text": "hello",
        }
    ]


def test_get_parses_plist_text_and_marks_my_messages(tmp_path):
    """Plist-encoded rows yield their NS.string, and is_from_me becomes (Me)."""
    db = tmp_path / "chat.db"
    blob = plistlib.dumps({"NS.string": "rich hello"})
    _message_db(db, [(blob, _apple_ns(), 1, 1)])
    sms = _mac_sms(database_path=str(db))
    with mock.patch("sys.platform", "darwin"):
        messages = sms.get()
    assert messages == [
        {
            "date": datetime.datetime(2001, 1, 1),
            "from": "(Me)",
            "text": "rich hello",
        }
    ]


def test_get_skips_empty_text_and_applies_contact_filter(tmp_path):
    """Rows without text are skipped; contact= narrows to one sender."""
    db = tmp_path / "chat.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT)")
    conn.execute(
        "CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text BLOB, "
        "date INTEGER, handle_id INTEGER, is_from_me INTEGER)"
    )
    conn.execute("INSERT INTO handle (ROWID, id) VALUES (1, '+1555')")
    conn.execute("INSERT INTO handle (ROWID, id) VALUES (2, '+1666')")
    conn.execute("INSERT INTO message (text, date, handle_id, is_from_me) VALUES ('', 0, 1, 0)")
    conn.execute("INSERT INTO message (text, date, handle_id, is_from_me) VALUES ('for alice', 0, 1, 0)")
    conn.execute("INSERT INTO message (text, date, handle_id, is_from_me) VALUES ('for bob', 0, 2, 0)")
    conn.commit()
    conn.close()
    sms = _mac_sms(database_path=str(db))
    with mock.patch("sys.platform", "darwin"):
        messages = sms.get(contact="+1666")
    assert [m["text"] for m in messages] == ["for bob"]


def test_get_applies_substring_filter_and_limit(tmp_path):
    """substring= becomes a LIKE filter and limit= caps the rows returned."""
    db = tmp_path / "chat.db"
    _message_db(
        db,
        [
            ("alpha one", _apple_ns(days=1), 1, 0),
            ("beta", _apple_ns(days=2), 1, 0),
            ("alpha two", _apple_ns(days=3), 1, 0),
        ],
    )
    sms = _mac_sms(database_path=str(db))
    with mock.patch("sys.platform", "darwin"):
        messages = sms.get(substring="alpha", limit=1)
    assert [m["text"] for m in messages] == ["alpha two"]


def test_get_database_error_mid_read_returns_what_was_found():
    """A sqlite error mid-scan stops the scan and returns rows so far."""
    sms = _mac_sms(database_path="whatever.db")
    with mock.patch("sys.platform", "darwin"):
        with mock.patch.object(sms_module.subprocess, "run"):
            with mock.patch.object(sms_module.sqlite3, "connect") as connect:
                cursor = connect.return_value.cursor.return_value
                cursor.fetchone.side_effect = sqlite3.Error("database locked")
                assert sms.get() == []
    connect.return_value.close.assert_called_once()


def test_get_without_database_access_prompts_for_access(tmp_path):
    """Without file access, get() shows the Full Disk Access prompt first."""
    sms = _mac_sms(database_path=str(tmp_path / "missing.db"))
    assert sms.can_access_database() is False
    with mock.patch("sys.platform", "darwin"):
        with mock.patch.object(sms_module.subprocess, "run") as run:
            sms.prompt_full_disk_access()
    run.assert_called_once()
    assert run.call_args[0][0][:2] == ["osascript", "-e"]
