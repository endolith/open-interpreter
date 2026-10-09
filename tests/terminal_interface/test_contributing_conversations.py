import pytest
import json
from unittest import mock

from interpreter.terminal_interface import contributing_conversations as cc


def test_is_list_of_lists():
    """is_list_of_lists is True only for lists whose elements are all lists."""
    assert cc.is_list_of_lists([[1], [2]])
    assert not cc.is_list_of_lists([1, 2])
    # vacuous truth: all([]) is True in Python
    assert cc.is_list_of_lists([])


def test_get_contribute_cache_contents_creates_default(tmp_path, monkeypatch):
    """get_contribute_cache_contents creates a default cache file when none exists."""
    cache_path = tmp_path / "contribute.json"
    monkeypatch.setattr(cc, "contribute_cache_path", str(cache_path))
    result = cc.get_contribute_cache_contents()
    assert cache_path.exists()

    # All three flags, not just one. The two "asked_to_contribute" flags default
    # to False, and a True default would mean the user is never asked for consent
    # to send past conversations or future ones — a silent opt-out that still
    # reads as a fresh cache.
    assert result == {
        "displayed_contribution_message": False,
        "asked_to_contribute_past": False,
        "asked_to_contribute_future": False,
    }


def test_write_to_contribution_cache_round_trip(tmp_path, monkeypatch):
    """write_to_contribution_cache persists payload that get_contribute_cache_contents can read."""
    cache_path = tmp_path / "contribute.json"
    monkeypatch.setattr(cc, "contribute_cache_path", str(cache_path))
    payload = {
        "displayed_contribution_message": True,
        "asked_to_contribute_past": True,
        "asked_to_contribute_future": False,
    }
    cc.write_to_contribution_cache(payload)
    with open(cache_path) as f:
        assert json.load(f) == payload


def test_get_all_conversations_reads_json_files(tmp_path):
    """get_all_conversations loads conversation lists from .json files only."""
    history = tmp_path / "history"
    history.mkdir()
    (history / "a.json").write_text(json.dumps([["msg"]]))
    (history / "b.txt").write_text("skip me")

    interpreter = mock.Mock()
    interpreter.conversation_history_path = str(history)

    conversations = cc.get_all_conversations(interpreter)
    assert len(conversations) == 1
    assert conversations[0] == [["msg"]]


def test_contribute_conversations_posts_payload():
    """contribute_conversations POSTs conversations, feedback, and id to the server."""
    conversations = [[{"role": "user", "content": "hi"}]]
    with mock.patch(
        "interpreter.terminal_interface.contributing_conversations.requests.post"
    ) as post:
        cc.contribute_conversations(conversations, feedback="good", conversation_id="abc")
    post.assert_called_once()
    payload = post.call_args.kwargs["json"]
    assert payload["conversation_id"] == "abc"
    assert payload["feedback"] == "good"
    assert payload["conversations"] == conversations


def test_contribute_conversations_skips_empty():
    """contribute_conversations does nothing when given an empty conversation list."""
    with mock.patch(
        "interpreter.terminal_interface.contributing_conversations.requests.post"
    ) as post:
        assert cc.contribute_conversations([]) is None
        post.assert_not_called()


def _clean_cache(**overrides):
    """A contribution cache with every flag False, plus any overrides."""
    cache = {
        "displayed_contribution_message": False,
        "asked_to_contribute_past": False,
        "asked_to_contribute_future": False,
    }
    cache.update(overrides)
    return cache


@pytest.mark.parametrize("response, expected", [("y", True), ("Y", True), ("yes", False), ("n", False), ("", False)])
def test_user_wants_to_contribute_past_is_true_only_for_y(response, expected):
    """The past-conversations prompt accepts only "y", case-insensitively.

    This is the gate before past conversations are uploaded, so the comparison
    must stay exact: `in` or a truthiness check would treat "yes" and "" as
    consent, and typing anything at a prompt should not send a transcript.
    """
    with mock.patch("builtins.input", return_value=response):
        assert cc.user_wants_to_contribute_past() is expected


@pytest.mark.parametrize("response, expected", [("y", True), ("Y", True), ("maybe", False), ("n", False)])
def test_user_wants_to_contribute_future_is_true_only_for_y(response, expected):
    """The future-conversations prompt accepts only "y", case-insensitively."""
    with mock.patch("builtins.input", return_value=response):
        assert cc.user_wants_to_contribute_future() is expected


def test_send_past_conversations_does_not_prompt_when_there_are_none():
    """With nothing to send, the user is never asked for permission.

    Prompts unconditionally would be a consent bug in its own right: the user
    answers a question about uploading their history when no upload is possible.
    """
    interpreter = mock.Mock()
    with mock.patch.object(cc, "get_all_conversations", return_value=[]):
        with mock.patch("builtins.input") as prompt:
            cc.send_past_conversations(interpreter)

    prompt.assert_not_called()


def test_send_past_conversations_uploads_only_after_consent():
    """A declined prompt sends nothing; an accepted one uploads the transcript."""
    interpreter = mock.Mock()
    conversations = [[{"role": "user", "content": "secret"}]]

    with mock.patch.object(cc, "get_all_conversations", return_value=conversations):
        with mock.patch("builtins.input", return_value="n"):
            with mock.patch.object(cc, "contribute_conversations") as send:
                with mock.patch.object(cc.time, "sleep"):
                    cc.send_past_conversations(interpreter)
    send.assert_not_called()

    with mock.patch.object(cc, "get_all_conversations", return_value=conversations):
        with mock.patch("builtins.input", return_value="y"):
            with mock.patch.object(cc, "contribute_conversations") as send:
                with mock.patch.object(cc.time, "sleep"):
                    cc.send_past_conversations(interpreter)
    send.assert_called_once_with(conversations)


def test_set_send_future_conversations_persists_the_opt_in():
    """Accepting future contributions writes the profile key and says so."""
    interpreter = mock.Mock()
    with mock.patch.object(cc, "write_key_to_profile") as write_key:
        with mock.patch.object(cc, "display_markdown_message"):
            cc.set_send_future_conversations(interpreter, True)

    write_key.assert_called_once_with("contribute_conversation", True)


def test_launch_logic_runs_the_past_and_future_flow_when_already_contributing():
    """will_contribute=True goes straight to the two prompts, skipping the pitch."""
    interpreter = mock.Mock(will_contribute=True)
    cache = _clean_cache()

    with mock.patch.object(cc, "get_contribute_cache_contents", return_value=cache):
        with mock.patch.object(cc, "contribute_past_and_future_logic") as flow:
            with mock.patch.object(cc, "display_contribution_message") as pitch:
                with mock.patch.object(cc, "write_to_contribution_cache") as write:
                    cc.contribute_conversation_launch_logic(interpreter)

    # Identity, not equality: the function mutates the cache dict it is handed,
    # so by the time this assertion runs the flag has already been flipped and a
    # value comparison could never match.
    assert flow.call_args.args[0] is interpreter
    assert flow.call_args.args[1] is cache
    pitch.assert_not_called()
    write.assert_called_once()


def test_launch_logic_pitches_once_then_records_that_it_did():
    """A user who has not opted in sees the message once, and never again.

    The cache flag is the only thing keeping this from recurring on every launch,
    so both the write and the flag it writes are asserted.
    """
    interpreter = mock.Mock(will_contribute=False)

    with mock.patch.object(cc, "contribute_past_and_future_logic") as flow:
        with mock.patch.object(cc, "get_contribute_cache_contents", return_value=_clean_cache()):
            with mock.patch.object(cc, "display_contribution_message") as pitch:
                with mock.patch.object(cc.time, "sleep"):
                    with mock.patch.object(cc, "write_to_contribution_cache") as write:
                        cc.contribute_conversation_launch_logic(interpreter)

    flow.assert_not_called()
    pitch.assert_called_once()
    assert write.call_args.args[0]["displayed_contribution_message"] is True


def test_past_and_future_logic_does_not_re_ask_questions_already_answered():
    """Flags already set suppress both prompts, whatever the user would answer."""
    interpreter = mock.Mock()
    cache = _clean_cache(asked_to_contribute_past=True, asked_to_contribute_future=True)

    with mock.patch("builtins.input") as prompt:
        with mock.patch.object(cc, "display_contributing_current_message") as current:
            cc.contribute_past_and_future_logic(interpreter, cache)

    prompt.assert_not_called()
    assert current.called


def test_past_and_future_logic_sends_history_when_the_user_agrees():
    """Answering "y" to the past prompt triggers the upload exactly once."""
    interpreter = mock.Mock()
    cache = _clean_cache()
    conversations = [[{"role": "user", "content": "hi"}]]

    # Three answers, not two. Agreeing to contribute past conversations only opts
    # in; send_past_conversations then asks its own separate permission question
    # before anything is uploaded. Collapsing these into one prompt would let a
    # single keystroke carry a transcript off the machine.
    with mock.patch("builtins.input", side_effect=["y", "y", "n"]):
        with mock.patch.object(cc, "get_all_conversations", return_value=conversations):
            with mock.patch.object(cc, "contribute_conversations") as send:
                with mock.patch.object(cc.time, "sleep"):
                    with mock.patch.object(cc, "display_contributing_current_message"):
                        cc.contribute_past_and_future_logic(interpreter, cache)

    send.assert_called_once_with(conversations)
    assert cache["asked_to_contribute_past"] is True


def test_past_and_future_logic_opt_in_to_the_future_persists_the_profile_key():
    """Answering "y" to the future prompt enables future contributions.

    Answered "n" to the past question, so this isolates the second branch.
    """
    interpreter = mock.Mock()
    cache = _clean_cache()

    with mock.patch("builtins.input", side_effect=["n", "y"]):
        with mock.patch.object(cc, "send_past_conversations"):
            with mock.patch.object(cc, "write_key_to_profile") as write_key:
                with mock.patch.object(cc, "display_markdown_message"):
                    with mock.patch.object(cc, "display_contributing_current_message"):
                        cc.contribute_past_and_future_logic(interpreter, cache)

    write_key.assert_called_once_with("contribute_conversation", True)
    assert cache["asked_to_contribute_future"] is True


def test_contribute_conversations_skips_a_conversation_with_no_messages():
    """An empty inner list is treated as nothing to send.

    The guard checks both the outer list and the first conversation's length, so
    a saved-but-empty transcript must not produce a POST.
    """
    with mock.patch(
        "interpreter.terminal_interface.contributing_conversations.requests.post"
    ) as post:
        assert cc.contribute_conversations([[]]) is None
        post.assert_not_called()


def test_contribute_conversations_swallows_a_failed_request():
    """A network failure is deliberately ignored so contributing never blocks.

    The bare except is intentional and commented as non-blocking: a user who has
    opted in should not have their conversation interrupted because the upload
    endpoint is unreachable. This pins that intent, since "tidy up the bare
    except" would otherwise look like a safe cleanup.
    """
    conversations = [[{"role": "user", "content": "hi"}]]
    with mock.patch(
        "interpreter.terminal_interface.contributing_conversations.requests.post",
        side_effect=ConnectionError("unreachable"),
    ) as post:
        assert cc.contribute_conversations(conversations) is None
    post.assert_called_once()


def test_send_past_conversations_warns_about_private_information(capsys):
    """The consent prompt states what is being sent, the privacy risk, and how to check.

    This is the notice that makes the upload defensible. Asserting the substance
    rather than the exact string keeps rewording possible — but dropping the
    warning, the "don't contain private information" caution, or the pointer to
    `interpreter --conversations` would mean a user consents to sending a
    transcript without being told what it contains or how to inspect it first.
    """
    conversations = [[{"role": "user", "content": "private"}]]
    interpreter = mock.Mock()

    with mock.patch.object(cc, "get_all_conversations", return_value=conversations):
        with mock.patch("builtins.input", return_value="n"):
            with mock.patch.object(cc.time, "sleep"):
                cc.send_past_conversations(interpreter)

    out = capsys.readouterr().out
    assert "private information" in out
    assert "--conversations" in out
    assert "previous conversations" in out
