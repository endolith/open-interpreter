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
    assert result["displayed_contribution_message"] is False
    assert cache_path.exists()


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
