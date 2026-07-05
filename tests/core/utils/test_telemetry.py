import json
from unittest import mock

from interpreter.core.utils import telemetry
from tests.helpers import patch_expanduser


def test_get_or_create_uuid_reads_existing(tmp_path, monkeypatch):
    """get_or_create_uuid returns the ID from disk when the cache file already exists."""
    patch_expanduser(monkeypatch, telemetry, tmp_path)
    cache_dir = tmp_path / ".cache" / "open-interpreter"
    cache_dir.mkdir(parents=True)
    uuid_file = cache_dir / "telemetry_user_id"
    uuid_file.write_text("existing-id")

    assert telemetry.get_or_create_uuid() == "existing-id"


def test_get_or_create_uuid_creates_new(tmp_path, monkeypatch):
    """get_or_create_uuid generates a new UUID4 and persists it when no cache file exists."""
    patch_expanduser(monkeypatch, telemetry, tmp_path)
    new_id = telemetry.get_or_create_uuid()
    uuid_file = tmp_path / ".cache" / "open-interpreter" / "telemetry_user_id"
    assert uuid_file.read_text() == new_id
    # Must be a valid UUID4 (8-4-4-4-12 hex digits separated by dashes)
    import re
    assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", new_id)


def test_send_telemetry_posts_event():
    """send_telemetry POSTs a JSON payload with the event name, properties, and oi_version."""
    with mock.patch("interpreter.core.utils.telemetry.requests.post") as post:
        telemetry.send_telemetry("test_event", {"foo": "bar"})
    post.assert_called_once()
    payload = json.loads(post.call_args.kwargs["data"])
    assert payload["event"] == "test_event"
    assert payload["properties"]["foo"] == "bar"
    assert "oi_version" in payload["properties"]
