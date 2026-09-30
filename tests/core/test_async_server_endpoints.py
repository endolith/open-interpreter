import json
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from interpreter.core.async_core import AsyncInterpreter, Server


@pytest.fixture
def server_pair():
    """Build a (TestClient, AsyncInterpreter) pair for a fresh server."""
    interpreter = AsyncInterpreter()
    return TestClient(Server(interpreter).app), interpreter


@pytest.fixture
def interpreter():
    """A fresh AsyncInterpreter for building ad-hoc test servers."""
    return AsyncInterpreter()


@pytest.fixture
def client(server_pair):
    return server_pair[0]


@pytest.fixture
def client_no_raise(interpreter):
    """TestClient that converts endpoint exceptions into 500 responses."""
    return TestClient(Server(interpreter).app, raise_server_exceptions=False)


@pytest.fixture
def insecure_pair(monkeypatch):
    """(TestClient, AsyncInterpreter) pair with the insecure routes registered.

    create_router() reads INTERPRETER_INSECURE_ROUTES at router build time, so
    the env var must be set before Server() is constructed.
    """
    monkeypatch.setenv("INTERPRETER_INSECURE_ROUTES", "true")
    interpreter = AsyncInterpreter()
    return TestClient(Server(interpreter).app), interpreter


def test_heartbeat_endpoint(client):
    """GET /heartbeat returns a simple aliveness payload."""
    response = client.get("/heartbeat")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_home_endpoint_serves_html(client):
    """GET / serves the HTML chat page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text


def test_post_input_returns_success(client):
    """POST / accepts an LMC chunk and reports success."""
    response = client.post("/", json={"role": "user", "type": "message", "start": True})
    assert response.status_code == 200
    assert response.json() == {"status": "success"}


def test_post_input_propagates_error(client, server_pair):
    """POST / returns a 500 with the error message when input() fails."""
    _, interpreter = server_pair
    interpreter.input = mock.AsyncMock(side_effect=ValueError("boom"))
    response = client.post("/", json={"role": "user", "type": "message", "start": True})
    assert response.status_code == 500
    assert response.json() == {"error": "boom"}


def test_get_setting_returns_serialized_value(client):
    """GET /settings/{name} returns the serialized interpreter attribute."""
    response = client.get("/settings/auto_run")
    assert response.status_code == 200
    assert json.loads(json.loads(response.text)) == {"auto_run": False}


def test_get_setting_unknown_name_returns_404(client):
    """GET /settings/{name} for an unknown name returns a 404 response."""
    response = client.get("/settings/no_such_setting")
    assert response.status_code == 404
    assert response.json() == {"error": "Setting not found"}


def test_post_settings_unknown_llm_subsetting_returns_404(client):
    """POST /settings with an unknown llm sub-key returns a 404 response."""
    response = client.post("/settings", json={"llm": {"no_such_subkey": True}})
    assert response.status_code == 404
    assert response.json() == {
        "error": "Sub-setting no_such_subkey not found in llm"
    }


def test_post_settings_unknown_top_level_key_returns_404(client):
    """POST /settings with an unknown top-level key returns a 404 response."""
    response = client.post("/settings", json={"no_such_setting": True})
    assert response.status_code == 404
    assert response.json() == {"error": "Setting no_such_setting not found"}


def test_chat_completion_rejects_non_user_last_message(client_no_raise):
    """The OpenAI-compatible endpoint requires the last message to be from the user."""
    response = client_no_raise.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "assistant", "content": "hi"}]},
    )
    assert response.status_code == 500


def test_chat_completion_stop_token(client, server_pair):
    """The {STOP} sentinel sets then clears the stop event without a response."""
    _, interpreter = server_pair
    with mock.patch("interpreter.core.async_core.time.sleep"):
        response = client.post(
            "/openai/chat/completions",
            json={"messages": [{"role": "user", "content": "{STOP}"}]},
        )
    assert response.status_code == 200
    assert not interpreter.stop_event.is_set()


def test_chat_completion_context_mode_on(client, server_pair):
    """{CONTEXT_MODE_ON} and {REQUIRE_START_ON} enable context mode."""
    _, interpreter = server_pair
    for token in ["{CONTEXT_MODE_ON}", "{REQUIRE_START_ON}"]:
        client.post(
            "/openai/chat/completions",
            json={"messages": [{"role": "user", "content": token}]},
        )
        assert interpreter.context_mode is True


def test_chat_completion_context_mode_off(client, server_pair):
    """{CONTEXT_MODE_OFF} and {REQUIRE_START_OFF} disable context mode."""
    _, interpreter = server_pair
    interpreter.context_mode = True
    for token in ["{CONTEXT_MODE_OFF}", "{REQUIRE_START_OFF}"]:
        client.post(
            "/openai/chat/completions",
            json={"messages": [{"role": "user", "content": token}]},
        )
        assert interpreter.context_mode is False


def test_chat_completion_auto_run_toggle(client, server_pair):
    """{AUTO_RUN_ON} / {AUTO_RUN_OFF} flip the auto_run flag."""
    _, interpreter = server_pair
    client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "{AUTO_RUN_ON}"}]},
    )
    assert interpreter.auto_run is True
    client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "{AUTO_RUN_OFF}"}]},
    )
    assert interpreter.auto_run is False


def test_chat_completion_text_message_returns_assistant_reply(client, server_pair):
    """A plain text user message is appended and answered via chat()."""
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "content": "Hello there"}]
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "Hello there"
    assert any(
        m.get("content") == "hello" for m in interpreter.messages
    ), "the user message should be stored"


def test_chat_completion_list_text_content(client, server_pair):
    """A content list with a text part is appended as a user message."""
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "content": "ok"}]
    )
    response = client.post(
        "/openai/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]}
            ]
        },
    )
    assert response.status_code == 200
    assert any(
        m.get("type") == "message" for m in interpreter.messages
    ), "the text part should become a message"


def test_chat_completion_list_base64_image(client, server_pair):
    """A content list with a base64 image becomes an image message."""
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "content": "ok"}]
    )
    url = "data:image/png;base64,iVBORw0KGgo="
    response = client.post(
        "/openai/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": url}}],
                }
            ]
        },
    )
    assert response.status_code == 200
    image_messages = [m for m in interpreter.messages if m.get("type") == "image"]
    assert len(image_messages) == 1
    assert image_messages[0]["format"] == "base64.png"
    assert image_messages[0]["content"] == "iVBORw0KGgo="


def test_chat_completion_image_url_without_url_raises(client_no_raise):
    """An image_url part missing the url field is rejected."""
    response = client_no_raise.post(
        "/openai/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
            ]
        },
    )
    assert response.status_code == 500


def test_chat_completion_image_url_without_base64_raises(client_no_raise):
    """An image_url that is not a base64 data URI is rejected."""
    response = client_no_raise.post(
        "/openai/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
                    ],
                }
            ]
        },
    )
    assert response.status_code == 500


def test_chat_completion_stream_run_code(client, server_pair):
    """Streaming a 'yes' after a code message streams chunks from _respond_and_store."""
    _, interpreter = server_pair
    interpreter.messages = [
        {
            "role": "assistant",
            "type": "code",
            "format": "python",
            "content": "print(1)",
        }
    ]
    interpreter._respond_and_store = mock.MagicMock(
        return_value=iter(
            [
                {"role": "assistant", "type": "message", "content": "hi"},
                {"role": "assistant", "type": "code", "start": True, "format": "python"},
                {"role": "assistant", "type": "code", "end": True, "format": "python"},
            ]
        )
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )
    assert response.status_code == 200
    assert "chat.completion.chunk" in response.text
    assert "hi" in response.text


def test_chat_completion_stream_confirmation_breaks(client, server_pair):
    """A confirmation chunk with auto_run off asks and stops streaming."""
    _, interpreter = server_pair
    interpreter.auto_run = False
    interpreter._respond_and_store = mock.MagicMock(
        return_value=iter(
            [
                {
                    "role": "computer",
                    "type": "confirmation",
                    "content": {"format": "python", "content": "print(1)"},
                }
            ]
        )
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )
    assert response.status_code == 200
    assert "Do you want to run this code?" in response.text


def test_chat_completion_stream_message_via_chat(client, server_pair):
    """Streaming a normal message streams chunks produced by chat()."""
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "type": "message", "content": "hi"}]
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )
    assert response.status_code == 200
    assert "hi" in response.text


def test_insecure_routes_not_registered_by_default(monkeypatch):
    """The /run route is absent unless INTERPRETER_INSECURE_ROUTES is enabled."""
    monkeypatch.delenv("INTERPRETER_INSECURE_ROUTES", raising=False)
    interpreter = AsyncInterpreter()
    client = TestClient(Server(interpreter).app)
    response = client.post("/run", json={"language": "python", "code": "1+1"})
    assert response.status_code == 404


def test_insecure_run_route(insecure_pair):
    """With INTERPRETER_INSECURE_ROUTES=true, /run executes code via the computer."""
    client, interpreter = insecure_pair
    interpreter.computer.run = mock.MagicMock(return_value="42")

    response = client.post("/run", json={"language": "python", "code": "1+1"})
    assert response.status_code == 200
    assert response.json() == {"output": "42"}


def test_insecure_run_route_requires_language_and_code(insecure_pair):
    """The /run route rejects requests missing language or code with a 400."""
    client, _ = insecure_pair

    response = client.post("/run", json={"language": "python"})
    assert response.status_code == 400
    assert response.json() == {
        "error": "Both 'language' and 'code' are required."
    }


def test_insecure_run_route_propagates_error(insecure_pair):
    """The /run route returns a 500 when computer.run raises."""
    client, interpreter = insecure_pair
    interpreter.computer.run = mock.MagicMock(side_effect=RuntimeError("oops"))

    response = client.post("/run", json={"language": "python", "code": "1+1"})
    assert response.status_code == 500
    assert response.json() == {"error": "oops"}


def test_insecure_upload_route(insecure_pair, tmp_path):
    """The /upload route writes the uploaded file to the requested path."""
    client, _ = insecure_pair
    target = tmp_path / "out.txt"

    response = client.post(
        "/upload",
        files={"file": ("in.txt", b"payload")},
        data={"path": str(target)},
    )
    assert response.status_code == 200
    assert target.read_text() == "payload"


def test_insecure_upload_route_propagates_error(insecure_pair):
    """The /upload route returns a 500 when it cannot write the file."""
    client, _ = insecure_pair

    response = client.post(
        "/upload",
        files={"file": ("in.txt", b"payload")},
        data={"path": "/no/such/dir/out.txt"},
    )
    assert response.status_code == 500
    assert "No such file or directory" in response.json()["error"]


def test_insecure_download_route(insecure_pair, tmp_path):
    """The /download route streams a file at an absolute path."""
    client, _ = insecure_pair
    source = tmp_path / "file.bin"
    source.write_bytes(b"xyz")

    response = client.get(f"/download/{source}")
    assert response.status_code == 200
    assert response.content == b"xyz"


def test_insecure_download_route_missing_file(insecure_pair):
    """The /download route returns a 500 for a missing file."""
    client, _ = insecure_pair

    response = client.get("/download/nope.bin")
    assert response.status_code == 500
    assert "No such file or directory" in response.json()["error"]
