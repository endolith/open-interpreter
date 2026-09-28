from interpreter.terminal_interface.utils.get_conversations import get_conversations


def test_get_conversations_lists_only_json_files(monkeypatch, tmp_path):
    """get_conversations returns only .json files from the conversations directory."""
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("ignore me")

    monkeypatch.setattr(
        "interpreter.terminal_interface.utils.get_conversations.get_storage_path",
        lambda subdirectory: tmp_path,
    )

    assert set(get_conversations()) == {"a.json", "b.json"}


def test_get_conversations_returns_empty_when_no_json(monkeypatch, tmp_path):
    """get_conversations returns an empty list when the directory has no JSON files."""
    (tmp_path / "notes.txt").write_text("ignore me")

    monkeypatch.setattr(
        "interpreter.terminal_interface.utils.get_conversations.get_storage_path",
        lambda subdirectory: tmp_path,
    )

    assert get_conversations() == []


def test_get_conversations_reads_the_conversations_subdirectory(monkeypatch, tmp_path):
    """get_conversations asks storage for the "conversations" subdirectory.

    The subdirectory name is what routes the listing to the right folder; a
    changed or dropped argument would list a different (or the root) directory.
    """
    seen = {}

    def fake_storage_path(subdirectory):
        seen["subdirectory"] = subdirectory
        return str(tmp_path)

    monkeypatch.setattr(
        "interpreter.terminal_interface.utils.get_conversations.get_storage_path",
        fake_storage_path,
    )

    get_conversations()

    assert seen["subdirectory"] == "conversations"
