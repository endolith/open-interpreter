"""Tests for the async server WebSocket endpoint and authentication layer.

The browser-facing WebSocket interface (websocket_endpoint/receive_input/
send_output) and the HTTP auth middleware (validate_api_key) are how external
clients actually talk to Open Interpreter, so a regression here breaks every
frontend integration at once. These tests exercise the full flow end-to-end
through TestClient.websocket_connect.
"""

import asyncio
import json
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from interpreter.core.async_core import AsyncInterpreter, Server, authenticate_function


async def _hang_output():
    """Stub for interpreter.output() that parks forever without spinning.

    send_output() polls output() continuously once connected; a stub that
    returns immediately would busy-loop and race the assertions below.
    asyncio.sleep keeps the task cancellable, so gather() unwinds cleanly
    when the test client disconnects.
    """
    await asyncio.sleep(3600)


@pytest.fixture
def interpreter():
    """A fresh AsyncInterpreter for the websocket flow tests."""
    return AsyncInterpreter()


@pytest.fixture
def client(interpreter, monkeypatch):
    """A TestClient bound to the interpreter's server app with auth open."""
    monkeypatch.delenv("INTERPRETER_API_KEY", raising=False)
    return TestClient(Server(interpreter).app)


@pytest.fixture
def ws_pair(client, interpreter):
    """(client, interpreter, input-mock) with output emission parked."""
    interpreter.output = _hang_output
    interpreter.require_acknowledge = False
    inp = mock.AsyncMock()
    interpreter.input = inp
    return client, interpreter, inp


def _handshake(ws):
    """Complete the mandatory auth exchange for a server with no API key set."""
    ws.send_text(json.dumps({"auth": "not-checked-without-api-key"}))
    assert ws.receive_json() == {"auth": True}


def test_heartbeat_bypasses_api_key_auth(monkeypatch, interpreter):
    """With an API key configured, /heartbeat must stay reachable without a key."""
    monkeypatch.setenv("INTERPRETER_API_KEY", "secret")
    client = TestClient(Server(interpreter).app)
    response = client.get("/heartbeat")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_requests_without_key_allowed_when_no_api_key_configured(monkeypatch, client):
    """No INTERPRETER_API_KEY means open access: requests pass with any/no header."""
    monkeypatch.delenv("INTERPRETER_API_KEY", raising=False)
    response = client.get("/settings/auto_run")
    assert response.status_code == 200


def test_missing_api_key_header_rejected(monkeypatch, client):
    """When INTERPRETER_API_KEY is set, requests without X-API-KEY get 403."""
    monkeypatch.setenv("INTERPRETER_API_KEY", "secret")
    response = client.get("/settings/auto_run")
    assert response.status_code == 403
    assert response.json() == {"detail": "Authentication failed"}


def test_wrong_api_key_rejected(monkeypatch, client):
    """A non-matching X-API-KEY value is rejected with 403."""
    monkeypatch.setenv("INTERPRETER_API_KEY", "secret")
    response = client.get("/settings/auto_run", headers={"X-API-KEY": "wrong"})
    assert response.status_code == 403


def test_correct_api_key_accepted(monkeypatch, client):
    """The matching X-API-KEY grants normal access to protected routes."""
    monkeypatch.setenv("INTERPRETER_API_KEY", "secret")
    response = client.get("/settings/auto_run", headers={"X-API-KEY": "secret"})
    assert response.status_code == 200


def test_authenticate_function_open_when_no_key(monkeypatch):
    """Without INTERPRETER_API_KEY, authenticate_function accepts everything."""
    monkeypatch.delenv("INTERPRETER_API_KEY", raising=False)
    assert authenticate_function(None) is True
    assert authenticate_function("anything") is True


def test_authenticate_function_requires_exact_match(monkeypatch):
    """With INTERPRETER_API_KEY set, only an equal string authenticates."""
    monkeypatch.setenv("INTERPRETER_API_KEY", "secret")
    assert authenticate_function("secret") is True
    assert authenticate_function("Secret") is False
    assert authenticate_function(None) is False


def test_host_and_port_setters_rebuild_uvicorn_server():
    """Assigning host/port updates uvicorn's config with a fresh server object.

    The setters intentionally recreate self.uvicorn_server so changes made
    after __init__ are picked up by run(); document that contract.
    """
    server = Server(AsyncInterpreter())
    old = server.uvicorn_server
    server.host = "127.0.0.3"
    assert server.host == "127.0.0.3"
    assert server.uvicorn_server is not old
    old = server.uvicorn_server
    server.port = 6001
    assert server.port == 6001
    assert server.uvicorn_server is not old


def test_foreign_origin_rejected_before_accept(client):
    """Browser origins off localhost are refused with policy code 1008."""
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/", headers={"Origin": "http://evil.example"}):
            pass  # pragma: no cover - connection should be refused
    assert excinfo.value.code == 1008
    assert excinfo.value.reason == "Origin not allowed"


def test_correct_auth_receives_confirmation(ws_pair):
    """The socket echoes {\"auth\": True} after an accepted credential."""
    client, _, _ = ws_pair
    with client.websocket_connect("/") as ws:
        _handshake(ws)


def test_wrong_credential_reports_failure_then_can_retry(ws_pair, monkeypatch):
    """Bad credentials yield {\"auth\": False}; a later good one still works."""
    client, _, _ = ws_pair
    monkeypatch.setenv("INTERPRETER_API_KEY", "secret")
    with client.websocket_connect("/") as ws:
        ws.send_text(json.dumps({"auth": "bad"}))
        assert ws.receive_json() == {"auth": False}
        ws.send_text(json.dumps({"auth": "secret"}))
        assert ws.receive_json() == {"auth": True}


def test_payload_before_authentication_is_discarded_not_processed(ws_pair):
    """LMC chunks sent pre-auth are swallowed (no input()), answered auth:False.

    KNOWN BUG-ish: payloads silently disappear instead of being queued until
    auth completes; clients unaware of the handshake lose their first message.
    Documenting current behavior.
    """
    client, _, inp = ws_pair
    with client.websocket_connect("/") as ws:
        ws.send_text(json.dumps({"role": "user", "start": True}))
        assert ws.receive_json() == {"auth": False}
    assert inp.await_count == 0


def test_authenticated_chunk_forwarded_parsed(ws_pair):
    """After the handshake, an LMC chunk reaches input() as the parsed dict."""
    client, _, inp = ws_pair
    with client.websocket_connect("/") as ws:
        _handshake(ws)
        ws.send_text(json.dumps({"role": "user", "start": True}))
    inp.assert_awaited_once_with({"role": "user", "start": True})


def test_binary_frame_forwarded_raw(ws_pair):
    """Bytes frames bypass JSON parsing and reach input() unchanged."""
    client, _, inp = ws_pair
    with client.websocket_connect("/") as ws:
        _handshake(ws)
        ws.send_bytes(b"\x00\x01binary")
    inp.assert_awaited_once_with(b"\x00\x01binary")


def test_acknowledgement_recorded_and_not_forwarded(
    ws_pair, monkeypatch
):
    """require_acknowledge turns {\"ack\": id} frames into receipt bookkeeping.

    They must be recorded in acknowledged_outputs and never reach input().
    """
    client, interp, inp = ws_pair
    interp.require_acknowledge = True
    interp.acknowledged_outputs.clear()
    with client.websocket_connect("/") as ws:
        _handshake(ws)
        ws.send_text(json.dumps({"ack": "msg-id-1"}))
    assert interp.acknowledged_outputs == ["msg-id-1"]
    assert inp.await_count == 0


def test_invalid_json_after_handshake_emits_server_error(ws_pair):
    """Garbage text frames produce a server error chunk + complete marker.

    The receive_input except branch formats the traceback and pushes both an
    error message and complete_message back over the socket while connected.
    """
    client, _, inp = ws_pair
    with client.websocket_connect("/") as ws:
        _handshake(ws)
        ws.send_text("{not valid json!!")
        error = ws.receive_json()
        done = ws.receive_json()
    assert error["type"] == "error"
    assert error["role"] == "server"
    assert "JSONDecodeError" in error["content"]
    assert done == {"role": "server", "type": "status", "content": "complete"}
    assert inp.await_count == 0
