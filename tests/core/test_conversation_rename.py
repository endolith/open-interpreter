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


def test_manual_rename_does_not_overwrite_another_conversation(
    interpreter_with_conversation_file,
):
    """A manual title colliding with another saved conversation is rejected."""
    oi = interpreter_with_conversation_file
    collision = os.path.join(
        oi.conversation_history_path,
        "Same_title__January_01_2025_12-00-00.json",
    )
    with open(collision, "w", encoding="utf-8") as f:
        f.write('{"messages": ["keep me"]}')

    assert not oi.rename_conversation_file_from_llm_title(manual_title="Same title")
    assert oi.conversation_filename == "untitled__January_01_2025_12-00-00.json"
    with open(collision, encoding="utf-8") as f:
        assert "keep me" in f.read()


def test_manual_rename_marks_title_upgraded(interpreter_with_conversation_file):
    """A manual rename opts the conversation out of later automatic retitling."""
    oi = interpreter_with_conversation_file

    assert oi.rename_conversation_file_from_llm_title(manual_title="My title")
    assert oi._conversation_title_upgraded is True


def test_auto_title_is_attempted_once_even_when_it_fails(
    interpreter_with_conversation_file,
):
    """A failed title request is not retried on every save."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.messages = [
        {"role": "user", "type": "message", "content": "First"},
        {"role": "assistant", "type": "message", "content": "One"},
        {"role": "user", "type": "message", "content": "Second"},
    ]
    final_path = os.path.join(oi.conversation_history_path, oi.conversation_filename)

    with mock.patch.object(
        oi, "_run_llm_for_conversation_title_slug", return_value=""
    ) as generate:
        oi._maybe_upgrade_conversation_title(final_path)
        oi._maybe_upgrade_conversation_title(final_path)

    assert generate.call_count == 1
    assert oi._conversation_title_upgraded is True


def test_auto_title_chooses_a_free_filename(interpreter_with_conversation_file):
    """Automatic rename must not overwrite a conversation with the same slug and time."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.messages = [
        {"role": "user", "type": "message", "content": "First"},
        {"role": "assistant", "type": "message", "content": "One"},
        {"role": "user", "type": "message", "content": "Second"},
    ]
    final_path = os.path.join(oi.conversation_history_path, oi.conversation_filename)
    collision = os.path.join(
        oi.conversation_history_path,
        "Auto_title__January_01_2025_12-00-00.json",
    )
    with open(collision, "w", encoding="utf-8") as f:
        f.write("{}")

    with mock.patch.object(
        oi, "_run_llm_for_conversation_title_slug", return_value="Auto_title"
    ):
        oi._maybe_upgrade_conversation_title(final_path)

    assert (
        oi.conversation_filename
        == "Auto_title__January_01_2025_12-00-00_1.json"
    )
    assert os.path.isfile(collision)


def test_run_llm_title_uses_plain_stream_and_skips_reasoning(
    interpreter_with_conversation_file,
):
    """Title generation ignores reasoning chunks and sanitizes the returned text."""
    oi = interpreter_with_conversation_file
    oi.messages = [{"role": "user", "type": "message", "content": "Hello"}]
    oi.llm = mock.Mock()
    oi.llm.run.return_value = [
        {"format": "reasoning", "content": "think"},
        {"content": "My title"},
    ]

    assert oi._run_llm_for_conversation_title_slug("User: Hello") == "My_title"
    oi.llm.run.assert_called_once()
    assert oi.llm.run.call_args.kwargs["auxiliary_title_request"] is True


def test_run_llm_title_returns_empty_on_provider_error(
    interpreter_with_conversation_file,
):
    """A provider failure yields no title instead of escaping the rename path."""
    oi = interpreter_with_conversation_file
    oi.llm = mock.Mock()
    oi.llm.run.side_effect = RuntimeError("provider down")

    assert oi._run_llm_for_conversation_title_slug("User: Hello") == ""


def test_maybe_upgrade_does_nothing_without_a_titleable_transcript(
    interpreter_with_conversation_file,
):
    """Auto-title is skipped when there are too few qualifying messages."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.messages = [{"role": "assistant", "type": "message", "content": "Only me"}]

    with mock.patch.object(oi, "_run_llm_for_conversation_title_slug") as generate:
        oi._maybe_upgrade_conversation_title(
            os.path.join(oi.conversation_history_path, oi.conversation_filename)
        )

    generate.assert_not_called()


def test_maybe_upgrade_does_nothing_for_legacy_filename(
    interpreter_with_conversation_file,
):
    """Auto-title requires the filename to retain its timestamp segment."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.conversation_filename = "untitled.json"
    oi.messages = [
        {"role": "user", "type": "message", "content": "First"},
        {"role": "user", "type": "message", "content": "Second"},
    ]

    with mock.patch.object(oi, "_run_llm_for_conversation_title_slug") as generate:
        oi._maybe_upgrade_conversation_title(
            os.path.join(oi.conversation_history_path, oi.conversation_filename)
        )

    generate.assert_not_called()


def test_run_llm_title_returns_empty_for_reasoning_only_output(
    interpreter_with_conversation_file,
):
    """A title completion with no content yields no title."""
    oi = interpreter_with_conversation_file
    oi.llm = mock.Mock()
    oi.llm.run.return_value = [{"format": "reasoning", "content": "thinking"}]

    assert oi._run_llm_for_conversation_title_slug("User: Hello") == ""


def test_rename_with_llm_title_rejects_empty_transcript(
    interpreter_with_conversation_file,
):
    """The LLM rename path requires something titleable in the transcript."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.messages = []

    assert not oi.rename_conversation_file_from_llm_title()
    assert oi.conversation_filename == "untitled__January_01_2025_12-00-00.json"


def test_rename_with_llm_title_rejects_empty_model_output(
    interpreter_with_conversation_file,
):
    """An empty model title leaves the conversation filename unchanged."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.messages = [{"role": "user", "type": "message", "content": "Hi"}]

    with mock.patch.object(oi, "_run_llm_for_conversation_title_slug", return_value=""):
        assert not oi.rename_conversation_file_from_llm_title()


def test_rename_legacy_filename_gets_a_timestamp(
    interpreter_with_conversation_file,
):
    """A filename without a timestamp segment gains one during rename."""
    oi = interpreter_with_conversation_file
    oi.conversation_filename = "untitled.json"
    old_path = os.path.join(oi.conversation_history_path, oi.conversation_filename)
    with open(old_path, "w", encoding="utf-8") as f:
        f.write("{}")

    with mock.patch.object(
        oi, "_sanitize_conversation_title_slug", return_value="Title"
    ):
        assert oi.rename_conversation_file_from_llm_title(manual_title="Title")

    assert oi.conversation_filename.startswith("Title__")
    assert oi.conversation_filename.endswith(".json")


def test_auto_title_skips_missing_filename(interpreter_with_conversation_file):
    """Auto-title is a no-op when no conversation filename is set."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.conversation_filename = None

    with mock.patch.object(oi, "_run_llm_for_conversation_title_slug") as generate:
        oi._maybe_upgrade_conversation_title("ignored.json")

    generate.assert_not_called()


def test_auto_title_skips_empty_transcript(interpreter_with_conversation_file):
    """Auto-title is a no-op when the transcript trims to nothing."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.messages = [
        {"role": "user", "type": "message", "content": "First"},
        {"role": "user", "type": "message", "content": "Second"},
    ]

    with mock.patch.object(oi, "_conversation_auto_title_transcript", return_value=""):
        with mock.patch.object(oi, "_run_llm_for_conversation_title_slug") as generate:
            oi._maybe_upgrade_conversation_title(
                os.path.join(oi.conversation_history_path, oi.conversation_filename)
            )

    generate.assert_not_called()


def test_auto_title_skips_when_generated_slug_matches_current(
    interpreter_with_conversation_file,
):
    """The auto-title path does not replace an already matching filename."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.messages = [
        {"role": "user", "type": "message", "content": "First"},
        {"role": "user", "type": "message", "content": "Second"},
    ]

    with mock.patch.object(
        oi, "_run_llm_for_conversation_title_slug", return_value="untitled"
    ):
        oi._maybe_upgrade_conversation_title(
            os.path.join(oi.conversation_history_path, oi.conversation_filename)
        )

    assert oi.conversation_filename == "untitled__January_01_2025_12-00-00.json"
    assert oi._conversation_title_upgraded is True


def test_streaming_chat_saves_and_titles_after_normal_completion(tmp_path):
    """A completed stream saves the JSON and then runs the title upgrade."""
    oi = OpenInterpreter(conversation_history=True, offline=False)
    oi.conversation_history_path = str(tmp_path)
    oi.conversation_filename = "untitled__January_01_2025_12-00-00.json"
    oi.display_message = lambda _message: None

    def respond():
        yield {"type": "message", "content": "reply"}

    oi._respond_and_store = respond
    oi._maybe_upgrade_conversation_title = mock.Mock()

    list(oi._streaming_chat(message="hi", display=False))

    path = tmp_path / oi.conversation_filename
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))
    oi._maybe_upgrade_conversation_title.assert_called_once_with(str(path))


def test_streaming_chat_saves_but_does_not_title_an_interrupted_stream(tmp_path):
    """Closing the stream early still saves, but skips the blocking title request."""
    oi = OpenInterpreter(conversation_history=True, offline=False)
    oi.conversation_history_path = str(tmp_path)
    oi.conversation_filename = "untitled__January_01_2025_12-00-00.json"
    oi.display_message = lambda _message: None

    def respond():
        yield {"type": "message", "content": "first"}
        yield {"type": "message", "content": "second"}

    oi._respond_and_store = respond
    oi._maybe_upgrade_conversation_title = mock.Mock()

    stream = oi._streaming_chat(message="hi", display=False)
    next(stream)
    stream.close()

    assert (tmp_path / oi.conversation_filename).is_file()
    oi._maybe_upgrade_conversation_title.assert_not_called()


def test_streaming_chat_cleans_up_temp_file_when_save_fails(tmp_path):
    """A failed atomic replace still removes the temporary file."""
    oi = OpenInterpreter(conversation_history=True, offline=False)
    oi.conversation_history_path = str(tmp_path)
    oi.conversation_filename = "untitled__January_01_2025_12-00-00.json"

    def respond():
        yield {"type": "message", "content": "reply"}

    oi._respond_and_store = respond

    with mock.patch(
        "interpreter.core.core.os.replace", side_effect=OSError("disk full")
    ):
        with pytest.raises(OSError, match="disk full"):
            list(oi._streaming_chat(message="hi", display=False))

    assert not list(tmp_path.glob("*.tmp"))


def test_streaming_chat_without_a_message_needs_display():
    """The private stream generator refuses a no-message call without display."""
    oi = OpenInterpreter(conversation_history=False)

    with pytest.raises(Exception, match="requires a display"):
        list(oi._streaming_chat(display=False))


def test_streaming_chat_derives_a_filename_from_a_single_word(tmp_path):
    """The one-word branch creates a safe name and the history directory."""
    history = tmp_path / "nested" / "conversations"
    oi = OpenInterpreter(conversation_history=True, offline=False)
    oi.conversation_history_path = str(history)
    oi.conversation_filename = None
    oi.display_message = lambda _message: None

    def respond():
        yield {"type": "message", "content": "reply"}

    oi._respond_and_store = respond

    list(oi._streaming_chat(message="Hello", display=False))

    assert oi.conversation_filename.startswith("Hello__")
    assert (history / oi.conversation_filename).is_file()


def test_streaming_chat_sanitizes_invalid_filename_characters(tmp_path):
    """The filename generator strips unsafe characters before saving."""
    oi = OpenInterpreter(conversation_history=True, offline=False)
    oi.conversation_history_path = str(tmp_path)
    oi.conversation_filename = None
    oi.display_message = lambda _message: None

    def respond():
        yield {"type": "message", "content": "reply"}

    oi._respond_and_store = respond

    list(oi._streaming_chat(message='a<>:"/\\|?*!\n b', display=False))

    assert oi.conversation_filename.startswith("a__")
    assert (tmp_path / oi.conversation_filename).is_file()


def test_run_llm_title_ignores_chunks_without_content(
    interpreter_with_conversation_file,
):
    """A title stream may contain metadata chunks; they are skipped."""
    oi = interpreter_with_conversation_file
    oi.llm = mock.Mock()
    oi.llm.run.return_value = [{"foo": "bar"}, {"content": "Title"}]

    assert oi._run_llm_for_conversation_title_slug("User: Hello") == "Title"


def test_streaming_chat_without_history_skips_save_and_title():
    """Disabling conversation history bypasses disk persistence and title upgrade."""
    oi = OpenInterpreter(conversation_history=False, offline=False)

    def respond():
        yield {"type": "message", "content": "reply"}

    oi._respond_and_store = respond
    oi._maybe_upgrade_conversation_title = mock.Mock()

    list(oi._streaming_chat(message="hi", display=False))

    oi._maybe_upgrade_conversation_title.assert_not_called()


def test_title_transcript_trim_keeps_both_ends():
    """Long transcripts keep the start, an omission marker, and the end."""
    body = "start " + ("x" * 500) + " end"
    from interpreter.core.core import _conversation_title_transcript_trim_to_cap

    result = _conversation_title_transcript_trim_to_cap(body, 200)
    assert result.startswith("start")
    assert result.endswith("end")
    assert "middle of conversation omitted" in result
    assert len(result) <= 200


def test_title_transcript_trim_handles_tiny_cap():
    """A cap too small for the omission marker falls back to the tail."""
    from interpreter.core.core import _conversation_title_transcript_trim_to_cap

    body = "a" * 50 + "tail"
    assert _conversation_title_transcript_trim_to_cap(body, 20) == body[-20:]


def test_title_transcript_clips_individual_messages():
    """Per-message clipping caps long code/tool dumps."""
    oi = OpenInterpreter()
    text = "x" * 3000
    clipped = oi._clip_conversation_title_text(text)
    assert clipped.startswith("x" * 2500)
    assert clipped.endswith("[…truncated…]")
    assert len(clipped) < len(text)


def test_auto_title_transcript_applies_total_cap():
    """The transcript builder honors an explicit total cap."""
    oi = OpenInterpreter()
    oi.messages = [
        {"role": "user", "type": "message", "content": "u" * 1000},
        {"role": "assistant", "type": "message", "content": "a" * 1000},
    ]

    transcript = oi._conversation_auto_title_transcript(total_char_cap=100)

    assert len(transcript) <= 100


def test_manual_rename_reports_os_error(interpreter_with_conversation_file):
    """A filesystem failure during %rename is reported, not raised."""
    oi = interpreter_with_conversation_file
    with mock.patch(
        "interpreter.core.core.os.replace", side_effect=OSError("locked")
    ):
        assert not oi.rename_conversation_file_from_llm_title(manual_title="Title")
    assert oi.conversation_filename == "untitled__January_01_2025_12-00-00.json"


def test_auto_title_ignores_os_error(interpreter_with_conversation_file):
    """A failed automatic rename is best-effort and does not fail the chat."""
    oi = interpreter_with_conversation_file
    oi.offline = False
    oi.messages = [
        {"role": "user", "type": "message", "content": "First"},
        {"role": "user", "type": "message", "content": "Second"},
    ]
    final_path = os.path.join(oi.conversation_history_path, oi.conversation_filename)
    with mock.patch.object(
        oi, "_run_llm_for_conversation_title_slug", return_value="Auto_title"
    ):
        with mock.patch(
            "interpreter.core.core.os.replace", side_effect=OSError("locked")
        ):
            oi._maybe_upgrade_conversation_title(final_path)
    assert oi.conversation_filename == "untitled__January_01_2025_12-00-00.json"
