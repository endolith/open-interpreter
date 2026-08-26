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


def test_send_output_does_not_spin_when_client_disconnects_with_unsent_messages():
    """A disconnected client with a stuck unsent message must not starve the event loop.

    Regression for the resume-connection flake in the authenticated server test:
    when a client drops mid-turn, send_message returns False without awaiting
    once the socket reports DISCONNECTED, so the drain loop in send_output used
    to spin forever (never popping the message, never yielding). That blocked
    the uvicorn event loop, stalling the next connection's WebSocket handshake
    until its open timeout fired.

    The fake WebSocket reports CONNECTED until the first send attempt, then
    flips to DISCONNECTED and raises, reproducing the drop-mid-send race. The
    endpoint coroutine must return promptly instead of hanging.

    The scenario runs in a subprocess with a hard timeout: the pre-fix spin is
    a tight GIL-bound loop that blocks the event loop (so an in-process
    asyncio.wait_for can never fire) and starves sibling threads (so even a
    thread.join budget is unreliable). A subprocess timeout is enforced by the
    OS and is immune to both.
    """

    import subprocess
    import sys

    script = f"""
import asyncio
import os
from collections import deque

os.environ["INTERPRETER_REQUIRE_AUTH"] = "False"

from interpreter.core.async_core import AsyncInterpreter, Server
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocketState
from websockets.exceptions import ConnectionClosedOK


async def run_endpoint():
    interpreter = AsyncInterpreter()
    interpreter.require_acknowledge = False
    interpreter.unsent_messages = deque(
        [{{"role": "assistant", "type": "message", "content": "pending"}}]
    )

    server = Server(interpreter)
    endpoint = next(
        r.endpoint
        for r in server.app.routes
        if isinstance(r, WebSocketRoute) and r.path == "/"
    )

    class DroppingWebSocket:
        \"\"\"Fake WebSocket that drops the connection on the first send.\"\"\"

        def __init__(self):
            self.headers = {{}}
            self.client_state = WebSocketState.CONNECTED

        async def accept(self):
            pass

        async def receive(self):
            return {{"type": "websocket.disconnect"}}

        async def send_text(self, data):
            self.client_state = WebSocketState.DISCONNECTED
            raise ConnectionClosedOK(None, None)

        async def send_bytes(self, data):
            self.client_state = WebSocketState.DISCONNECTED
            raise ConnectionClosedOK(None, None)

    await endpoint(DroppingWebSocket())


asyncio.run(run_endpoint())
print("COMPLETED")
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            "send_output spun the event loop on a disconnected client with "
            "unsent messages instead of returning"
        )

    assert result.returncode == 0, (
        f"endpoint subprocess failed: rc={result.returncode} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "COMPLETED" in result.stdout, (
        "endpoint subprocess did not report completion "
        f"(stdout={result.stdout!r} stderr={result.stderr!r})"
    )
