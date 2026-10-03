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


def test_send_telemetry_swallows_errors():
    """send_telemetry must not raise when the HTTP request fails (telemetry is non-blocking)."""
    with mock.patch(
        "interpreter.core.utils.telemetry.requests.post",
        side_effect=Exception("network"),
    ):
        telemetry.send_telemetry("test_event")


def test_send_telemetry_payload_shape():
    """The POST body carries the exact PostHog fields the endpoint expects.

    api_key, event, properties, and distinct_id are the capture contract; a
    renamed or dropped field would silently stop all telemetry while the
    existing assertions (event/properties only) still passed.
    """
    with mock.patch("interpreter.core.utils.telemetry.requests.post") as post:
        telemetry.send_telemetry("shape_event", {"k": "v"})

    payload = json.loads(post.call_args.kwargs["data"])
    assert set(payload) == {"api_key", "event", "properties", "distinct_id"}
    assert payload["api_key"].startswith("phc_")
    assert payload["distinct_id"] == telemetry.user_id


def test_send_telemetry_posts_to_posthog_with_json_headers():
    """The request targets the PostHog capture URL with a JSON content type.

    A wrong URL or missing Content-Type would make PostHog reject the payload;
    the URL and headers were never asserted.
    """
    with mock.patch("interpreter.core.utils.telemetry.requests.post") as post:
        telemetry.send_telemetry("url_event")

    args, kwargs = post.call_args
    assert args[0] == "https://app.posthog.com/capture"
    assert kwargs["headers"] == {"Content-Type": "application/json"}


def test_send_telemetry_defaults_properties_to_empty_dict():
    """Omitting properties sends an empty object that still gains oi_version.

    The default has to be a fresh dict; a None or list default would raise on
    the oi_version assignment or send the wrong JSON type.
    """
    with mock.patch("interpreter.core.utils.telemetry.requests.post") as post:
        telemetry.send_telemetry("no_props")

    payload = json.loads(post.call_args.kwargs["data"])
    assert payload["properties"]["oi_version"] == telemetry.version("open-interpreter")


def test_send_telemetry_merges_caller_properties_without_dropping_version():
    """Caller properties are preserved alongside the injected oi_version.

    The version is added into the caller's dict; a mutation that replaced the
    dict instead of updating it would discard the caller's fields.
    """
    with mock.patch("interpreter.core.utils.telemetry.requests.post") as post:
        telemetry.send_telemetry("merge_event", {"a": 1, "b": 2})

    props = json.loads(post.call_args.kwargs["data"])["properties"]
    assert props["a"] == 1
    assert props["b"] == 2
    assert "oi_version" in props


def test_get_or_create_uuid_falls_back_to_idk_on_error(tmp_path, monkeypatch):
    """An unwritable cache path falls back to the literal "idk" instead of raising.

    Telemetry is non-blocking, so a filesystem error must yield the sentinel id
    rather than propagate; the bare except returning "idk" is the whole guard.
    """
    patch_expanduser(monkeypatch, telemetry, tmp_path)
    # Point HOME at a path under a file so makedirs fails.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    monkeypatch.setattr(
        telemetry.os.path, "expanduser", lambda path: str(blocker / "home")
    )

    assert telemetry.get_or_create_uuid() == "idk"

