import os
from unittest import TestCase, mock

from interpreter.core.async_core import (
    AsyncInterpreter,
    Server,
    confirmation_digest,
    is_websocket_origin_allowed,
    SENSITIVE_LLM_SETTINGS,
    SENSITIVE_SERVER_SETTINGS,
)


class TestServerConstruction(TestCase):
    """
    Tests to make sure that the underlying server is configured correctly when constructing
    the Server object.
    """

    def test_host_and_port_defaults(self):
        """
        Tests that a Server object takes on the default host and port when
        a) no host and port are passed in, and
        b) no HOST and PORT are set.
        """
        with mock.patch.dict(os.environ, {}):
            s = Server(AsyncInterpreter())
            self.assertEqual(s.host, Server.DEFAULT_HOST)
            self.assertEqual(s.port, Server.DEFAULT_PORT)

    def test_host_and_port_passed_in(self):
        """
        Tests that a Server object takes on the passed-in host and port when they are passed-in,
        ignoring the surrounding HOST and PORT env vars.
        """
        host = "the-really-real-host"
        port = 2222

        with mock.patch.dict(
            os.environ,
            {"INTERPRETER_HOST": "this-is-supes-fake", "INTERPRETER_PORT": "9876"},
        ):
            sboth = Server(AsyncInterpreter(), host, port)
            self.assertEqual(sboth.host, host)
            self.assertEqual(sboth.port, port)

    def test_host_and_port_from_env_1(self):
        """
        Tests that the Server object takes on the HOST and PORT env vars as host and port when
        nothing has been passed in.
        """
        fake_host = "fake_host"
        fake_port = 1234

        with mock.patch.dict(
            os.environ,
            {"INTERPRETER_HOST": fake_host, "INTERPRETER_PORT": str(fake_port)},
        ):
            s = Server(AsyncInterpreter())
            self.assertEqual(s.host, fake_host)
            self.assertEqual(s.port, fake_port)


class TestWebSocketOriginPolicy(TestCase):
    def test_missing_origin_allowed_for_local_clients(self):
        self.assertTrue(is_websocket_origin_allowed(None))
        self.assertTrue(is_websocket_origin_allowed(""))

    def test_localhost_origins_allowed(self):
        self.assertTrue(is_websocket_origin_allowed("http://127.0.0.1:8000"))
        self.assertTrue(is_websocket_origin_allowed("http://localhost:8000"))

    def test_remote_origins_rejected(self):
        self.assertFalse(is_websocket_origin_allowed("https://evil.example"))

    def test_null_origin_allowed_for_non_browser_clients(self):
        """The literal 'null' origin is accepted, as some local clients send it."""
        self.assertTrue(is_websocket_origin_allowed("null"))

    def test_non_http_scheme_rejected_even_for_local_host(self):
        """Origins with a non-http scheme (e.g. ftp) are never allowed."""
        self.assertFalse(is_websocket_origin_allowed("ftp://localhost"))
        self.assertFalse(is_websocket_origin_allowed("file:///etc/passwd"))


class TestSettingsEndpointGuards(TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        self.client = TestClient(Server(AsyncInterpreter()).app)

    def _assert_settings_blocked(self, payload, error_substring):
        response = self.client.post("/settings", json=payload)
        self.assertEqual(response.status_code, 403)
        self.assertIn(error_substring, response.json()["error"])

    def test_post_settings_blocks_sensitive_server_attributes(self):
        """POST /settings must reject top-level keys that control execution or history."""
        for key in SENSITIVE_SERVER_SETTINGS:
            with self.subTest(key=key):
                self._assert_settings_blocked({key: True}, key)

    def test_post_settings_blocks_sensitive_llm_attributes(self):
        """POST /settings must reject llm.api_key and llm.api_base."""
        for sub_key in SENSITIVE_LLM_SETTINGS:
            with self.subTest(sub_key=sub_key):
                self._assert_settings_blocked(
                    {"llm": {sub_key: "secret"}}, f"llm.{sub_key}"
                )

    def test_post_settings_allows_non_sensitive_llm_model(self):
        """Non-sensitive llm fields like model remain writable via POST /settings."""
        response = self.client.post("/settings", json={"llm": {"model": "gpt-4o-mini"}})
        self.assertEqual(response.status_code, 200)

class TestAsyncApprovalBinding(TestCase):
    def setUp(self):
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False

    def test_confirmation_digest_is_stable(self):
        payload = {"type": "code", "format": "python", "content": "print('hi')"}
        self.assertEqual(
            confirmation_digest(payload),
            confirmation_digest(payload),
        )

    def test_approve_pending_confirmation_requires_pending_state(self):
        self.assertFalse(self.interpreter._approve_pending_confirmation())

    def test_approve_pending_confirmation_accepts_matching_digest(self):
        payload = {"type": "code", "format": "python", "content": "print(1)"}
        self.interpreter.pending_confirmation = payload
        self.interpreter.pending_confirmation_digest = confirmation_digest(payload)

        digest = self.interpreter.pending_confirmation_digest
        self.assertTrue(self.interpreter._approve_pending_confirmation(digest))
        self.assertTrue(self.interpreter._approval_granted)

    def test_approve_pending_confirmation_rejects_wrong_digest(self):
        payload = {"type": "code", "format": "python", "content": "print(1)"}
        self.interpreter.pending_confirmation = payload
        self.interpreter.pending_confirmation_digest = confirmation_digest(payload)

        self.assertFalse(self.interpreter._approve_pending_confirmation("deadbeef"))


class TestAsyncInputCommandHandling(TestCase):
    def setUp(self):
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.interpreter.output_queue = mock.MagicMock()
        self.interpreter.output_queue.sync_q = mock.MagicMock()
        self.interpreter.respond_thread = mock.MagicMock()
        self.interpreter.respond_thread.is_alive.return_value = True

    def _run_go_command(self, command="go"):
        import asyncio

        async def run():
            await self.interpreter.input(
                {"role": "user", "type": "command", "start": True}
            )
            await self.interpreter.input(
                {"role": "user", "type": "command", "content": command}
            )
            await self.interpreter.input(
                {"role": "user", "type": "command", "end": True}
            )

        asyncio.run(run())

    def _error_messages(self):
        return [
            call.args[0].get("content", "")
            for call in self.interpreter.output_queue.sync_q.put.call_args_list
            if call.args[0].get("type") == "error"
        ]

    def test_command_start_does_not_require_join(self):
        import asyncio

        async def run():
            await self.interpreter.input(
                {"role": "user", "type": "command", "start": True}
            )
            self.interpreter.respond_thread.join.assert_not_called()

        asyncio.run(run())

    def test_go_command_without_pending_confirmation_emits_error(self):
        """go with no pending code must not start a new respond thread."""
        self._run_go_command("go")
        self.assertIn("No pending code approval", self._error_messages()[0])

    def test_go_command_with_wrong_digest_emits_error(self):
        """go:<digest> must match the pending confirmation payload."""
        payload = {"type": "code", "format": "python", "content": "print(1)"}
        self.interpreter.pending_confirmation = payload
        self.interpreter.pending_confirmation_digest = confirmation_digest(payload)

        self._run_go_command("go:deadbeef")
        self.assertIn("No pending code approval", self._error_messages()[0])

    def test_go_command_with_matching_digest_and_live_thread_returns(self):
        """go:<digest> resumes the paused respond thread instead of spawning another."""
        payload = {"type": "code", "format": "python", "content": "print(1)"}
        digest = confirmation_digest(payload)
        self.interpreter.pending_confirmation = payload
        self.interpreter.pending_confirmation_digest = digest
        self.interpreter.respond_thread.is_alive.return_value = True

        self._run_go_command(f"go:{digest}")

        self.assertEqual(self._error_messages(), [])
        self.interpreter.respond_thread.start.assert_not_called()

    def test_go_command_with_matching_digest_but_dead_thread_emits_error(self):
        """go:<digest> after the respond thread exited cannot resume execution."""
        payload = {"type": "code", "format": "python", "content": "print(1)"}
        digest = confirmation_digest(payload)
        self.interpreter.pending_confirmation = payload
        self.interpreter.pending_confirmation_digest = digest
        self.interpreter.respond_thread.is_alive.return_value = False

        self._run_go_command(f"go:{digest}")

        self.assertIn("No active response is waiting", self._error_messages()[0])


class TestAsyncRespondApproval(TestCase):
    def setUp(self):
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.mock_q = mock.MagicMock()
        self.interpreter.output_queue = mock.MagicMock(sync_q=self.mock_q)

    def _confirmation_payload(self):
        return {"format": "python", "content": "print(1)"}

    def _run_respond_with_chunks(self, chunks, approve=True):
        import threading
        import time

        def fake_store():
            yield from chunks

        with mock.patch.object(self.interpreter, "_respond_and_store", fake_store):

            def respond_thread():
                self.interpreter.respond()

            worker = threading.Thread(target=respond_thread)
            worker.start()
            time.sleep(0.05)
            self.interpreter._approval_granted = approve
            self.interpreter._approval_event.set()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())

    def test_respond_waits_for_approval_then_continues(self):
        """With auto_run off, confirmation chunks pause until go approves the digest."""
        confirmation = {
            "type": "confirmation",
            "role": "computer",
            "content": self._confirmation_payload(),
        }
        console = {
            "type": "console",
            "role": "computer",
            "format": "output",
            "content": "ok",
        }

        self._run_respond_with_chunks([confirmation, console], approve=True)

        put_chunks = [call.args[0] for call in self.mock_q.put.call_args_list]
        confirmation_puts = [c for c in put_chunks if c.get("type") == "confirmation"]
        self.assertEqual(len(confirmation_puts), 1)
        self.assertEqual(
            confirmation_puts[0]["confirmation_id"],
            confirmation_digest(self._confirmation_payload()),
        )
        self.assertTrue(any(c.get("content") == "ok" for c in put_chunks))

    def test_respond_stops_when_approval_denied(self):
        """Denied approval must not run code after the confirmation chunk."""
        confirmation = {
            "type": "confirmation",
            "role": "computer",
            "content": self._confirmation_payload(),
        }
        console = {
            "type": "console",
            "role": "computer",
            "format": "output",
            "content": "ok",
        }

        self._run_respond_with_chunks([confirmation, console], approve=False)

        put_chunks = [call.args[0] for call in self.mock_q.put.call_args_list]
        self.assertFalse(any(c.get("content") == "ok" for c in put_chunks))


class TestServerRunAndSetters(TestCase):
    def test_host_setter_recreates_uvicorn_server(self):
        """Assigning Server.host rebuilds the underlying uvicorn.Server."""
        s = Server(AsyncInterpreter())
        old_uvicorn = s.uvicorn_server
        s.host = "0.0.0.0"
        self.assertEqual(s.host, "0.0.0.0")
        self.assertIsNot(s.uvicorn_server, old_uvicorn)
        self.assertEqual(s.uvicorn_server.config.host, "0.0.0.0")

    def test_port_setter_recreates_uvicorn_server(self):
        """Assigning Server.port rebuilds the underlying uvicorn.Server."""
        s = Server(AsyncInterpreter())
        old_uvicorn = s.uvicorn_server
        s.port = 9999
        self.assertEqual(s.port, 9999)
        self.assertIsNot(s.uvicorn_server, old_uvicorn)
        self.assertEqual(s.uvicorn_server.config.port, 9999)

    def test_run_warns_when_host_is_0_0_0_0(self):
        """run() with 0.0.0.0 warns about LAN exposure and binds to the public IP."""
        import contextlib
        import io

        s = Server(AsyncInterpreter())
        s.uvicorn_server.run = mock.Mock()
        # Set host directly on config (bypasses the property setter which recreates uvicorn).
        s.config.host = "0.0.0.0"
        buf = io.StringIO()
        # The 0.0.0.0 branch probes the LAN IP via a UDP socket; mock it so the
        # test never makes a real network call.
        fake_socket = mock.Mock()
        fake_socket.getsockname.return_value = ("192.168.1.50", 12345)
        with mock.patch(
            "interpreter.core.async_core.socket.socket", return_value=fake_socket
        ):
            with contextlib.redirect_stdout(buf):
                s.run()
        out = buf.getvalue()
        self.assertIn("Warning", out)
        self.assertIn("0.0.0.0", out)
        self.assertIn("192.168.1.50", out)
        s.uvicorn_server.run.assert_called_once_with()

    def test_run_with_explicit_host_and_port(self):
        """run(host, port) overrides the stored config and binds to that host."""
        import contextlib
        import io

        s = Server(AsyncInterpreter())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # Both setters rebuild the uvicorn server from the config, so capture
            # the replacement instance and assert THAT one is started.
            with mock.patch(
                "interpreter.core.async_core.uvicorn.Server"
            ) as uvicorn_server_cls:
                s.run(host="127.0.0.1", port=8080)
        self.assertEqual(s.host, "127.0.0.1")
        self.assertEqual(s.port, 8080)
        out = buf.getvalue()
        self.assertIn("127.0.0.1", out)
        self.assertIn("8080", out)
        uvicorn_server_cls.return_value.run.assert_called_once_with()


class TestServerAuthMiddleware(TestCase):
    def test_heartbeat_skips_authentication(self):
        """The /heartbeat route bypasses API-key auth even when a key is set."""
        with mock.patch.dict(
            os.environ, {"INTERPRETER_API_KEY": "supersecret"}
        ):
            from fastapi.testclient import TestClient

            client = TestClient(Server(AsyncInterpreter()).app)
            response = client.get("/heartbeat")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "alive")

    def test_wrong_api_key_returns_403(self):
        """An incorrect X-API-KEY is rejected with 403 by the auth middleware."""
        with mock.patch.dict(
            os.environ, {"INTERPRETER_API_KEY": "supersecret"}
        ):
            from fastapi.testclient import TestClient

            client = TestClient(Server(AsyncInterpreter()).app)
            response = client.post(
                "/settings",
                json={"llm": {"model": "gpt-4o-mini"}},
                headers={"X-API-KEY": "wrong"},
            )
            self.assertEqual(response.status_code, 403)
            self.assertEqual(
                response.json()["detail"], "Authentication failed"
            )

    def test_correct_api_key_returns_200(self):
        """The matching X-API-KEY is accepted by the auth middleware."""
        with mock.patch.dict(
            os.environ, {"INTERPRETER_API_KEY": "supersecret"}
        ):
            from fastapi.testclient import TestClient

            client = TestClient(Server(AsyncInterpreter()).app)
            response = client.post(
                "/settings",
                json={"llm": {"model": "gpt-4o-mini"}},
                headers={"X-API-KEY": "supersecret"},
            )
            self.assertEqual(response.status_code, 200)
