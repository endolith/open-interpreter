import json
import os
from unittest import mock

import pytest

from interpreter import OpenInterpreter


@pytest.fixture
def interpreter_with_conversation_file(tmp_path):
    oi = OpenInterpreter(conversation_history=True, offline=True)
    oi.conversation_history_path = str(tmp_path)
    oi.conversation_filename = "untitled__January_01_2025_12-00-00.json"
    oi.display_message = lambda _message: None
    path = tmp_path / oi.conversation_filename
    path.write_text(json.dumps({"messages": []}), encoding="utf-8")
    return oi


def test_sanitize_conversation_title_slug():
    """Slug sanitization strips unsafe filename characters and collapses whitespace."""
    oi = OpenInterpreter()
    assert oi._sanitize_conversation_title_slug("Some title here!") == "Some_title_here"
    assert oi._sanitize_conversation_title_slug("Git repo: packaging/branches") == (
        "Git_repo_packagingbranches"
    )


def test_conversation_auto_title_transcript_skips_terminal_alerts():
    """Title transcript includes user/assistant turns but not terminal system alerts."""
    oi = OpenInterpreter()
    oi.messages = [
        {"role": "user", "type": "message", "content": "Hello"},
        {"role": "assistant", "type": "message", "content": "Hi there"},
        {
            "role": "user",
            "type": "message",
            "content": "ignored",
            "format": "system_alert",
        },
        {"role": "user", "type": "message", "content": "ignored", "source": "terminal"},
    ]

    transcript = oi._conversation_auto_title_transcript()

    assert "User: Hello" in transcript
    assert "Assistant: Hi there" in transcript
    assert "ignored" not in transcript


def test_rename_with_manual_title(interpreter_with_conversation_file):
    """Manual %rename titles rename the on-disk JSON atomically."""
    oi = interpreter_with_conversation_file
    old_path = os.path.join(oi.conversation_history_path, oi.conversation_filename)

    assert oi.rename_conversation_file_from_llm_title(manual_title="Some title here!")

    assert oi.conversation_filename == "Some_title_here__January_01_2025_12-00-00.json"
    assert not os.path.isfile(old_path)
    assert os.path.isfile(
        os.path.join(oi.conversation_history_path, oi.conversation_filename)
    )


def test_rename_with_empty_manual_title_rejected(interpreter_with_conversation_file):
    """Invalid manual titles must not rename the conversation file."""
    oi = interpreter_with_conversation_file

    assert not oi.rename_conversation_file_from_llm_title(manual_title="<>:")

    assert oi.conversation_filename == "untitled__January_01_2025_12-00-00.json"


def test_rename_rejected_when_history_disabled():
    """rename_conversation_file_from_llm_title is a no-op without conversation_history."""
    oi = OpenInterpreter(conversation_history=False)
    oi.display_message = lambda _message: None

    assert not oi.rename_conversation_file_from_llm_title(manual_title="Title")


def test_rename_rejected_when_no_conversation_file_set():
    """rename requires an existing conversation_filename from a prior save."""
    oi = OpenInterpreter(conversation_history=True, offline=True)
    oi.conversation_filename = None
    oi.display_message = lambda _message: None

    assert not oi.rename_conversation_file_from_llm_title(manual_title="Title")


def test_rename_rejected_when_file_missing_on_disk(interpreter_with_conversation_file):
    """rename must fail if the JSON path was never written."""
    oi = interpreter_with_conversation_file
    os.remove(
        os.path.join(oi.conversation_history_path, oi.conversation_filename)
    )

    assert not oi.rename_conversation_file_from_llm_title(manual_title="Title")


def test_rename_rejected_when_slug_unchanged(interpreter_with_conversation_file):
    """Sanitized manual title matching the current prefix must not rename."""
    oi = interpreter_with_conversation_file
    oi.conversation_filename = "untitled__January_01_2025_12-00-00.json"

    assert not oi.rename_conversation_file_from_llm_title(manual_title="untitled")


def test_rename_with_llm_title(interpreter_with_conversation_file):
    """LLM rename path uses the transcript and _run_llm_for_conversation_title_slug."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.messages = [
        {"role": "user", "type": "message", "content": "Plan a trip"},
        {"role": "assistant", "type": "message", "content": "Sure"},
    ]

    with mock.patch.object(
        oi, "_run_llm_for_conversation_title_slug", return_value="Trip_planning"
    ):
        assert oi.rename_conversation_file_from_llm_title()

    assert oi.conversation_filename == "Trip_planning__January_01_2025_12-00-00.json"


def test_rename_with_llm_title_rejected_offline(interpreter_with_conversation_file):
    """LLM rename is unavailable in offline mode."""
    oi = interpreter_with_conversation_file
    oi.messages = [{"role": "user", "type": "message", "content": "Hi"}]

    assert not oi.rename_conversation_file_from_llm_title()


def test_maybe_upgrade_conversation_title_after_enough_user_messages(
    interpreter_with_conversation_file,
):
    """Auto-title runs once enough real user messages exist after a save."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.messages = [
        {"role": "user", "type": "message", "content": "First"},
        {"role": "assistant", "type": "message", "content": "One"},
        {"role": "user", "type": "message", "content": "Second"},
    ]
    final_path = os.path.join(
        oi.conversation_history_path, oi.conversation_filename
    )

    with mock.patch.object(
        oi, "_run_llm_for_conversation_title_slug", return_value="Auto_title"
    ):
        oi._maybe_upgrade_conversation_title(final_path)

    assert oi.conversation_filename == "Auto_title__January_01_2025_12-00-00.json"
    assert oi._conversation_title_upgraded is True
