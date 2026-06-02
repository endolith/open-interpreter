import json
import os

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
    oi = OpenInterpreter()
    assert oi._sanitize_conversation_title_slug("Some title here!") == "Some_title_here"
    assert oi._sanitize_conversation_title_slug('Git repo: packaging/branches') == (
        "Git_repo_packagingbranches"
    )


def test_rename_with_manual_title(interpreter_with_conversation_file):
    oi = interpreter_with_conversation_file
    old_path = os.path.join(oi.conversation_history_path, oi.conversation_filename)

    assert oi.rename_conversation_file_from_llm_title(manual_title="Some title here!")

    assert oi.conversation_filename == "Some_title_here__January_01_2025_12-00-00.json"
    assert not os.path.isfile(old_path)
    assert os.path.isfile(
        os.path.join(oi.conversation_history_path, oi.conversation_filename)
    )


def test_rename_with_empty_manual_title_rejected(interpreter_with_conversation_file):
    oi = interpreter_with_conversation_file

    assert not oi.rename_conversation_file_from_llm_title(manual_title="<>:")

    assert oi.conversation_filename == "untitled__January_01_2025_12-00-00.json"
