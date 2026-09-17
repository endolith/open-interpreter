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

import conftest
from interpreter import OpenInterpreter


def _logged_paths():
    """Drain the registry, returning it to a pristine state."""
    paths = list(conftest._test_interpreters)
    del conftest._test_interpreters[:]
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
    assert conftest._conversation_file(interpreter) is None


def test_conversation_file_none_when_no_filename(tmp_path):
    """Interpreters that never chatted (no filename) report no log path."""
    interpreter = _interpreter(filename=None)
    assert conftest._conversation_file(interpreter) is None


def test_conversation_file_rejects_non_json():
    """Non-.json paths are never removed, even if reported."""
    interpreter = _interpreter(filename="notes.txt")
    assert conftest._conversation_file(interpreter) is None


def test_conversation_file_returns_exact_reported_path(tmp_path):
    """The reported path is history dir joined with the interpreter's filename."""
    interpreter = _interpreter(filename="my_chat.json", history_path=str(tmp_path))
    assert conftest._conversation_file(interpreter) == os.path.join(
        str(tmp_path), "my_chat.json"
    )


def test_teardown_removes_reported_log(tmp_path):
    """A saved log at the reported path is deleted by teardown."""
    target = tmp_path / "saved.json"
    target.write_text("{}")
    interpreter = _interpreter(filename="saved.json", history_path=str(tmp_path))
    conftest._remove_test_conversations([interpreter])
    assert not target.exists()


def test_teardown_preserves_foreign_files(tmp_path):
    """Files no interpreter reported survive teardown untouched."""
    foreign = tmp_path / "user_chat.json"
    foreign.write_text("{}")
    conftest._remove_test_conversations([_interpreter(history=False)])
    assert foreign.exists()
    assert foreign.read_text() == "{}"


def test_teardown_tolerates_missing_files_and_errors(tmp_path, monkeypatch):
    """Missing paths and removal errors never fail teardown."""
    interpreter = _interpreter(
        filename="gone.json", history_path=str(tmp_path / "no_such_dir")
    )
    conftest._remove_test_conversations([interpreter])

    target = tmp_path / "locked.json"
    target.write_text("{}")
    locked = _interpreter(filename="locked.json", history_path=str(tmp_path))

    def _boom(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(conftest.os, "remove", _boom)
    try:
        conftest._remove_test_conversations([locked])
    finally:
        monkeypatch.undo()
    assert target.exists()


def test_constructed_interpreters_are_registered():
    """OpenInterpreter() inside a test lands in the teardown registry."""
    before = list(conftest._test_interpreters)
    try:
        interpreter = OpenInterpreter()
        assert interpreter in conftest._test_interpreters
    finally:
        for registered in list(conftest._test_interpreters):
            if registered not in before:
                conftest._test_interpreters.remove(registered)


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
    conftest._remove_test_conversations([interpreter])
    assert not saved.exists()
    _logged_paths()
