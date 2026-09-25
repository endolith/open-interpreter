"""Regression tests for the conversation-log cleanup fixture in conftest.py.

The autouse `_clean_test_conversations` fixture deletes files tests write
to the real conversations folder. Ownership comes from the writer object
itself — each registered interpreter reports its exact log path — rather
than filesystem observation, so foreign files can never be mistaken for
test artifacts. Each test leaves the registry empty so tests stay
order-independent.
"""

import os
from types import SimpleNamespace
from unittest import mock

from interpreter import OpenInterpreter
from tests.helpers import (
    _conversation_file,
    _remove_test_conversations,
    _snapshot_dir,
    _test_interpreters,
    begin_test_conversations,
    end_test_conversations,
)


def _logged_paths():
    """Drain the registry, returning it to a pristine state."""
    paths = list(_test_interpreters)
    del _test_interpreters[:]
    return paths


def _interpreter(history=True, filename="probe.json", history_path=None, **kwargs):
    """A lightweight stand-in carrying the three attributes teardown reads."""
    return SimpleNamespace(
        conversation_history=history,
        conversation_filename=filename,
        conversation_history_path=(
            history_path if history_path is not None else os.path.join("x", "y")
        ),
        **kwargs,
    )


def test_conversation_file_none_when_history_off(tmp_path):
    """History-disabled interpreters report no log path."""
    interpreter = _interpreter(history=False)
    assert _conversation_file(interpreter) is None


def test_conversation_file_none_when_no_filename(tmp_path):
    """Interpreters that never chatted (no filename) report no log path."""
    interpreter = _interpreter(filename=None)
    assert _conversation_file(interpreter) is None


def test_conversation_file_rejects_non_json():
    """Non-.json paths are never removed, even if reported."""
    interpreter = _interpreter(filename="notes.txt")
    assert _conversation_file(interpreter) is None


def test_conversation_file_returns_exact_reported_path(tmp_path):
    """The reported path is history dir joined with the interpreter's filename."""
    interpreter = _interpreter(filename="my_chat.json", history_path=str(tmp_path))
    assert _conversation_file(interpreter) == os.path.join(
        str(tmp_path), "my_chat.json"
    )


def test_teardown_removes_reported_log(tmp_path):
    """A saved log at the reported path is deleted by teardown."""
    target = tmp_path / "saved.json"
    target.write_text("{}")
    interpreter = _interpreter(filename="saved.json", history_path=str(tmp_path))
    _remove_test_conversations([interpreter])
    assert not target.exists()


def test_teardown_preserves_foreign_files(tmp_path):
    """Files no interpreter reported survive teardown untouched."""
    foreign = tmp_path / "user_chat.json"
    foreign.write_text("{}")
    _remove_test_conversations([_interpreter(history=False)])
    assert foreign.exists()
    assert foreign.read_text() == "{}"


def test_teardown_tolerates_missing_files_and_errors(tmp_path, monkeypatch):
    """Missing paths and removal errors never fail teardown."""
    interpreter = _interpreter(
        filename="gone.json", history_path=str(tmp_path / "no_such_dir")
    )
    _remove_test_conversations([interpreter])

    target = tmp_path / "locked.json"
    target.write_text("{}")
    locked = _interpreter(filename="locked.json", history_path=str(tmp_path))

    def _boom(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "remove", _boom)
    try:
        _remove_test_conversations([locked])
    finally:
        monkeypatch.undo()
    assert target.exists()


def test_constructed_interpreters_are_registered():
    """OpenInterpreter() inside a test lands in the teardown registry."""
    before = list(_test_interpreters)
    try:
        interpreter = OpenInterpreter()
        assert interpreter in _test_interpreters
    finally:
        for registered in list(_test_interpreters):
            if registered not in before:
                _test_interpreters.remove(registered)


def test_conversation_file_rejects_traversal(tmp_path):
    """Filenames escaping the history dir report no log path.

    os.path.join() returns absolute filenames unchanged, so without a
    containment check a filename like ../user.json would resolve outside
    the history directory and teardown could delete it.
    """
    interpreter = _interpreter(filename="../user.json", history_path=str(tmp_path))
    assert _conversation_file(interpreter) is None
    interpreter = _interpreter(
        filename=os.path.join(str(tmp_path), "other.json"),
        history_path=str(tmp_path / "sub"),
    )
    assert _conversation_file(interpreter) is None


def test_teardown_preserves_configured_but_unsaved_file(tmp_path):
    """A reported path for a file that predates the test is preserved.

    Configuring conversation_filename alone creates nothing — only chat()
    saves. If that file already existed before the test ran, teardown must
    leave it alone even though an interpreter reports it.
    """
    target = tmp_path / "existing.json"
    target.write_text("{}")
    before = _snapshot_dir(str(tmp_path))
    interpreter = _interpreter(filename="existing.json", history_path=str(tmp_path))
    _remove_test_conversations([interpreter], before)
    assert target.exists()
    assert target.read_text() == "{}"


def test_snapshot_failure_removes_nothing(tmp_path, monkeypatch):
    """A failed directory listing disables cleanup instead of emptying it.

    _snapshot_dir returns None (not an empty set) when os.listdir raises,
    and teardown with a None snapshot removes nothing — otherwise a
    PermissionError on listing would look like "no pre-existing files" and
    every reported path would be deleted.
    """
    target = tmp_path / "saved.json"
    target.write_text("{}")
    interpreter = _interpreter(filename="saved.json", history_path=str(tmp_path))

    removed = []
    orig_remove = os.remove

    def _recording_remove(path):
        removed.append(path)
        return orig_remove(path)

    monkeypatch.setattr(os, "remove", _recording_remove)
    try:
        _remove_test_conversations(
            [interpreter], None, os.path.realpath(str(tmp_path))
        )
    finally:
        monkeypatch.undo()
    assert removed == []
    assert target.exists()


def test_snapshot_failure_returns_none_sentinel(tmp_path, monkeypatch):
    """_snapshot_dir returns None (not empty set) when listing fails."""

    def _boom(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "listdir", _boom)
    try:
        assert _snapshot_dir(str(tmp_path)) is None
    finally:
        monkeypatch.undo()


def test_teardown_restricted_to_snapshot_directory(tmp_path):
    """Reported paths outside the snapshotted dir are preserved.

    Interpreters configured with custom history dirs report paths the
    default-directory snapshot never saw. Restricting removal to the
    snapshotted directory keeps those files — pytest already manages
    tmp_path cleanup itself.
    """
    custom = tmp_path / "custom"
    custom.mkdir()
    target = custom / "saved.json"
    target.write_text("{}")
    other = tmp_path / "real"
    other.mkdir()
    interpreter = _interpreter(filename="saved.json", history_path=str(custom))
    _remove_test_conversations(
        [interpreter], frozenset(), os.path.realpath(str(other))
    )
    assert target.exists()

    interpreter = _interpreter(filename="saved.json", history_path=str(custom))
    _remove_test_conversations(
        [interpreter], frozenset(), os.path.realpath(str(custom))
    )
    assert not target.exists()


def test_tracking_installed_at_import_time():
    """The __init__ wrapper is process-wide, not per-test.

    The tracking wrapper must already be in place before any test module is
    imported — otherwise module-level instances built at collection time
    (e.g. tests/test_interpreter.py) would never register and their files
    would never be cleaned. functools.wraps marks the wrapper with
    __wrapped__; a per-test monkeypatch without it would fail this test.
    """
    assert hasattr(OpenInterpreter.__init__, "__wrapped__")


def test_real_chat_save_is_cleaned_up(tmp_path):
    """End to end: a real chat() save is removed by the registry teardown."""
    interpreter = OpenInterpreter()
    interpreter.conversation_history = True
    interpreter.conversation_history_path = str(tmp_path)
    interpreter.conversation_filename = "e2e.json"

    with mock.patch.object(interpreter, "_respond_and_store", return_value=iter([])):
        interpreter.chat(message="hello", display=False)

    saved = tmp_path / "e2e.json"
    assert saved.exists()
    _remove_test_conversations([interpreter])
    assert not saved.exists()
    _logged_paths()


def test_shared_instance_cleaned_on_second_cycle(tmp_path, monkeypatch):
    """A pre-registered interpreter reused after a teardown is still cleaned.

    Regression: teardown used to drain the whole registry, so a module-level
    instance forgotten after cycle 1 left its cycle-2 file behind. Two
    simulated fixture cycles share one pre-registered instance; both saves
    are removed while the registration itself survives. Runs against an
    isolated directory so no real user file is ever touched.
    """
    monkeypatch.setattr(
        "tests.helpers._conversations_dir", lambda: str(tmp_path)
    )
    name = "__test_shared_cycle_probe__.json"
    target = os.path.join(str(tmp_path), name)
    shared = SimpleNamespace(
        conversation_history=True,
        conversation_filename=name,
        conversation_history_path=str(tmp_path),
    )
    _test_interpreters.append(shared)
    try:
        for _ in range(2):
            state = begin_test_conversations()
            with open(target, "w") as handle:
                handle.write("{}")
            end_test_conversations(state)
            assert not os.path.exists(target)
        assert shared in _test_interpreters
    finally:
        if shared in _test_interpreters:
            _test_interpreters.remove(shared)
        if os.path.exists(target):
            os.remove(target)
