import json
import os

import pytest

from interpreter import OpenInterpreter
from interpreter.core.core import _CONVERSATION_TITLE_TRANSCRIPT_OMITTED_MARKER
from interpreter.core.core import _conversation_title_transcript_trim_to_cap


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
    """Verifies model/LLM titles are turned into Windows-safe filename segments."""
    oi = OpenInterpreter()
    assert oi._sanitize_conversation_title_slug("Some title here!") == "Some_title_here"
    assert oi._sanitize_conversation_title_slug('Git repo: packaging/branches') == (
        "Git_repo_packagingbranches"
    )


def test_conversation_auto_title_transcript_orders_turns():
    """The transcript given to the title model must list User:/Assistant: turns in
    oldest-first order, skipping terminal-injected alerts and empty assistant turns."""
    oi = OpenInterpreter()
    oi.messages = [
        {"role": "assistant", "content": "", "type": "message"},
        {"role": "user", "content": "first question", "type": "message"},
        {"role": "user", "content": "injected alert", "type": "message", "source": "terminal"},
        {"role": "assistant", "content": "first answer", "type": "message"},
        {"role": "assistant", "content": "review", "type": "review"},
        {"role": "user", "content": "follow up", "type": "message"},
    ]
    body = oi._conversation_auto_title_transcript(total_char_cap=100000)
    assert body == (
        "User: first question\n\nAssistant: first answer\n\nUser: follow up"
    )


def test_conversation_auto_title_transcript_trims_long_chunks():
    """Very long single messages must be clipped so code or tool dumps cannot
    dominate the title prompt."""
    oi = OpenInterpreter()
    oi.messages = [
        {"role": "user", "content": "x" * 5000, "type": "message"},
    ]
    body = oi._conversation_auto_title_transcript(total_char_cap=100000)
    assert "\n[…truncated…]" in body
    assert body.startswith("User: " + "x" * 2500)


def test_conversation_title_transcript_keeps_head_and_tail_over_cap():
    """When the whole transcript exceeds the cap, the excerpt must keep the start and
    the end (so late-shifted topics still surface) with an explicit omission marker."""
    marker = _CONVERSATION_TITLE_TRANSCRIPT_OMITTED_MARKER
    cap = 200
    body = "begin " + ("middle " * 100) + "end"
    out = _conversation_title_transcript_trim_to_cap(body, cap)
    assert len(out) <= cap
    assert out.startswith("begin ")
    assert out.endswith("end")
    assert marker in out


def test_conversation_title_transcript_unchanged_under_cap():
    """Transcripts already within the cap must pass through untouched (no marker)."""
    body = "short transcript"
    assert _conversation_title_transcript_trim_to_cap(body, 100) == body


def test_rename_with_manual_title(interpreter_with_conversation_file):
    """A user-supplied title becomes the filename prefix (after slug sanitization)."""
    oi = interpreter_with_conversation_file
    old_path = os.path.join(oi.conversation_history_path, oi.conversation_filename)

    assert oi.rename_conversation_file_from_llm_title(manual_title="Some title here!")

    assert oi.conversation_filename == "Some_title_here__January_01_2025_12-00-00.json"
    assert not os.path.isfile(old_path)
    assert os.path.isfile(
        os.path.join(oi.conversation_history_path, oi.conversation_filename)
    )


def test_rename_with_empty_manual_title_rejected(interpreter_with_conversation_file):
    """A title that sanitizes to nothing must be rejected, keeping the file untouched."""
    oi = interpreter_with_conversation_file

    assert not oi.rename_conversation_file_from_llm_title(manual_title="<>:")

    assert oi.conversation_filename == "untitled__January_01_2025_12-00-00.json"
