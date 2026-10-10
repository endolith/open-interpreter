"""Tests for the async server WebSocket endpoint and authentication layer.

The browser-facing WebSocket interface (websocket_endpoint/receive_input/
send_output) and the HTTP auth middleware (validate_api_key) are how external
clients actually talk to Open Interpreter, so a regression here breaks every
frontend integration at once. These tests exercise the full flow end-to-end
through TestClient.websocket_connect.
"""

import asyncio
import json
import time
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


def test_payload_before_authentication_is_queued_then_delivered(ws_pair):
    """LMC chunks sent pre-auth are queued and delivered after auth (issue #249).

    The client is still told each pre-auth frame is unauthenticated, but the
    payload is held and reaches input() once the handshake completes instead
    of being silently lost.
    """
    client, _, inp = ws_pair
    with client.websocket_connect("/") as ws:
        ws.send_text(json.dumps({"role": "user", "start": True}))
        assert ws.receive_json() == {"auth": False}
        _handshake(ws)
    inp.assert_awaited_once_with({"role": "user", "start": True})


def test_payload_before_authentication_never_delivered_without_auth(ws_pair):
    """Queued pre-auth payloads are dropped with the connection if auth never completes."""
    client, _, inp = ws_pair
    with client.websocket_connect("/") as ws:
        ws.send_text(json.dumps({"role": "user", "start": True}))
        assert ws.receive_json() == {"auth": False}
    assert inp.await_count == 0


def test_binary_payload_before_authentication_is_queued_then_delivered(ws_pair):
    """Binary frames sent pre-auth are queued and delivered raw after auth (issue #249)."""
    client, _, inp = ws_pair
    with client.websocket_connect("/") as ws:
        ws.send_bytes(b"\x00\x01binary")
        assert ws.receive_json() == {"auth": False}
        _handshake(ws)
    inp.assert_awaited_once_with(b"\x00\x01binary")


def test_failed_auth_attempt_is_never_queued(ws_pair, monkeypatch):
    """A rejected credential is answered, not queued: after a later success only real payloads arrive."""
    client, _, inp = ws_pair
    monkeypatch.setenv("INTERPRETER_API_KEY", "secret")
    with client.websocket_connect("/") as ws:
        ws.send_text(json.dumps({"auth": "bad"}))
        assert ws.receive_json() == {"auth": False}
        ws.send_text(json.dumps({"role": "user", "start": True}))
        assert ws.receive_json() == {"auth": False}
        ws.send_text(json.dumps({"auth": "secret"}))
        assert ws.receive_json() == {"auth": True}
    inp.assert_awaited_once_with({"role": "user", "start": True})


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


def _emit(interpreter, *messages):
    """Replace output() so it yields `messages` then parks, in order.

    Returns a gate the test opens after the handshake. Emission has to be held
    back until then: send_output is gathered alongside receive_input, so a stub
    that produced immediately would deliver its payload *before* the auth reply,
    and every assertion downstream would be reading the wrong frame.

    The gate polls with a sleep rather than blocking, so the task stays
    cancellable and the client can still disconnect.
    """
    pending = list(messages)
    gate = {"open": False}

    async def output():
        while not gate["open"]:
            await asyncio.sleep(0.001)
        if pending:
            return pending.pop(0)
        await asyncio.sleep(3600)

    interpreter.output = output
    return gate


def test_output_reaches_a_connected_client(ws_pair):
    """The send side delivers interpreter output to an authenticated socket.

    The receive side is covered elsewhere; this is the other half of the same
    endpoint, and without it a chat turn produces nothing on the client no matter
    how correctly the input was parsed.
    """
    client, interp, _ = ws_pair
    gate = _emit(interp, {"role": "assistant", "type": "message", "content": "hello"})

    with client.websocket_connect("/") as ws:
        _handshake(ws)
        gate["open"] = True
        got = ws.receive_json()

    assert got == {"role": "assistant", "type": "message", "content": "hello"}


def test_bytes_output_is_sent_as_a_binary_frame(ws_pair):
    """A bytes output is framed with send_bytes, not JSON-encoded.

    Binary payloads (screenshots, audio) would otherwise be serialised as a
    string of escaped characters and arrive corrupt.
    """
    client, interp, _ = ws_pair
    gate = _emit(interp, b"\x89PNG\r\n\x1a\nrawbytes")

    with client.websocket_connect("/") as ws:
        _handshake(ws)
        gate["open"] = True
        got = ws.receive_bytes()

    assert got == b"\x89PNG\r\n\x1a\nrawbytes"


def test_acknowledged_output_gets_an_id_and_waits_for_the_ack(ws_pair):
    """With require_acknowledge on, the server assigns an id and withholds delivery.

    The client is expected to echo `{"ack": id}`; nothing is delivered until it
    does. Both halves matter — the id has to be attached to the payload, and the
    ack has to be what releases it.
    """
    client, interp, _ = ws_pair
    interp.require_acknowledge = True
    interp.acknowledged_outputs.clear()
    gate = _emit(interp, {"role": "assistant", "type": "message", "content": "hi"})

    # The list is instrumented so a removal is *observable*. Polling for
    # emptiness is not enough: the list is empty before the ack is recorded, so a
    # naive poll passes immediately whether or not the ack was ever processed —
    # and, as CodeRabbit pointed out, whether or not the removal ever ran.
    removals = []
    real_list = interp.acknowledged_outputs

    class RecordingList(list):
        def remove(self, value, *args, **kwargs):
            removals.append(value)
            return super().remove(value, *args, **kwargs)

    interp.acknowledged_outputs = RecordingList(real_list)

    with client.websocket_connect("/") as ws:
        _handshake(ws)
        gate["open"] = True
        payload = ws.receive_json()
        assert payload["id"], "an un-acked output must be given an id"

        ws.send_text(json.dumps({"ack": payload["id"]}))

        # Wait for the removal rather than asserting instantly. Sending the ack
        # only records the id; `send_message` notices it on its next pass through
        # a 0.1ms poll loop and removes it. Asserting the moment the ack is
        # written races that and loses on a loaded runner — it passed locally
        # every time and failed on both CI versions. Exiting the socket first is
        # worse: that cancels the task before it can observe the ack at all.
        for _ in range(150):
            if removals:
                break
            time.sleep(0.02)

    assert payload["id"] in removals, (
        f"the acknowledged id should be consumed by send_message; removals={removals}"
    )
    assert interp.acknowledged_outputs == [], (
        "a satisfied ack is consumed, not retained"
    )




def test_an_unacknowledged_output_is_queued_for_retry(ws_pair):
    """An output the client never acks is kept, not dropped.

    send_message gives up after its wait window and reports failure, and
    send_output then parks the message in `unsent_messages` to be retried. If it
    were discarded instead, a slow or briefly-disconnected client would silently
    lose assistant output — which is exactly the case acknowledgement exists to
    protect.
    """
    client, interp, _ = ws_pair
    interp.require_acknowledge = True
    interp.acknowledged_outputs.clear()
    interp.unsent_messages.clear()
    gate = _emit(interp, {"role": "assistant", "type": "message", "content": "unacked"})

    with client.websocket_connect("/") as ws:
        _handshake(ws)
        gate["open"] = True
        ws.receive_json()  # receive it; deliberately never ack

        # Poll while still connected. Closing the socket cancels the background
        # task, and send_message has a ~10ms ack window to elapse first, so
        # asserting after the `with` block observes a cancelled coroutine that
        # never reached its append.
        for _ in range(100):
            if interp.unsent_messages:
                break
            time.sleep(0.02)

    assert any(
        isinstance(m, dict) and m.get("content") == "unacked"
        for m in interp.unsent_messages
    ), f"expected the unacked output to be retained, got {interp.unsent_messages}"
