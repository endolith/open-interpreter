import asyncio
import json
from unittest import mock

import pytest

from interpreter.core.async_core import AsyncInterpreter, Server, complete_message


@pytest.fixture
def server_client():
    """FastAPI TestClient for a fresh AsyncInterpreter server."""
    from fastapi.testclient import TestClient

    return TestClient(Server(AsyncInterpreter()).app)


def test_post_settings_updates_llm_model(server_client):
    """POST /settings accepts non-sensitive llm fields such as model."""
    response = server_client.post("/settings", json={"llm": {"model": "gpt-4o-mini"}})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_post_settings_updates_top_level_verbose(server_client):
    """POST /settings accepts ordinary top-level interpreter flags."""
    response = server_client.post("/settings", json={"verbose": True})
    assert response.status_code == 200


def test_post_settings_updates_nested_llm_flag(server_client):
    """POST /settings accepts nested llm fields that are not credentials."""
    response = server_client.post(
        "/settings", json={"llm": {"supports_functions": False}}
    )
    assert response.status_code == 200


def test_get_settings_messages(server_client):
    """GET /settings/messages returns the conversation message list."""
    response = server_client.get("/settings/messages")
    assert response.status_code == 200
    payload = json.loads(response.text)
    assert "messages" in payload


def test_server_registers_websocket_route():
    """The async server exposes a WebSocket handler at /."""
    server = Server(AsyncInterpreter())
    paths = {getattr(route, "path", None) for route in server.app.routes}
    assert "/" in paths


def test_server_authenticate_accepts_any_key_when_env_unset(monkeypatch):
    """Server auth stays permissive when INTERPRETER_API_KEY is not configured."""
    monkeypatch.delenv("INTERPRETER_API_KEY", raising=False)
    server = Server(AsyncInterpreter())
    assert server.authenticate("any-key") is True


def test_respond_auto_run_streams_past_confirmation():
    """With auto_run on, respond() streams past confirmation and emits complete."""
    interpreter = AsyncInterpreter()
    interpreter.auto_run = True
    mock_q = mock.MagicMock()
    interpreter.output_queue = mock.MagicMock(sync_q=mock_q)

    chunks = [
        {
            "type": "confirmation",
            "role": "computer",
            "content": {"format": "python", "content": "print(1)"},
        },
        {
            "type": "console",
            "role": "computer",
            "format": "output",
            "content": "1",
        },
    ]

    def fake_respond_and_store():
        yield from chunks

    with mock.patch.object(interpreter, "_respond_and_store", fake_respond_and_store):
        interpreter.respond()

    put_chunks = [call.args[0] for call in mock_q.put.call_args_list]
    assert any(c.get("content") == "1" for c in put_chunks)
    assert put_chunks[-1] == complete_message


def test_input_user_message_end_starts_respond_thread():
    """Ending a user message turn starts respond() on a background thread."""
    async_i = AsyncInterpreter()
    async_i.messages = []

    with mock.patch("interpreter.core.async_core.threading.Thread") as thread_cls:
        asyncio.run(async_i.input({"role": "user", "type": "message", "start": True}))
        asyncio.run(
            async_i.input({"role": "user", "type": "message", "content": "hello"})
        )
        asyncio.run(async_i.input({"role": "user", "type": "message", "end": True}))

    thread_cls.assert_called_once()
    assert thread_cls.call_args.kwargs["target"] == async_i.respond


def test_input_stop_command_joins_active_respond_thread():
    """The stop command joins an active respond thread and sets stop_event."""
    async_i = AsyncInterpreter()
    mock_thread = mock.Mock()
    async_i.respond_thread = mock_thread
    async_i.messages = [{"role": "user", "type": "command", "content": "stop"}]

    asyncio.run(async_i.input({"role": "user", "type": "message", "end": True}))

    mock_thread.join.assert_called_once()
    assert async_i.stop_event.is_set()


def test_input_message_start_interrupts_active_respond_thread():
    """A new user message start interrupts an in-flight respond thread."""
    async_i = AsyncInterpreter()
    mock_thread = mock.Mock()
    mock_thread.is_alive.return_value = True
    async_i.respond_thread = mock_thread
    async_i.messages = []

    asyncio.run(async_i.input({"role": "user", "type": "message", "start": True}))

    mock_thread.join.assert_called_once()
    assert async_i.stop_event.is_set()
