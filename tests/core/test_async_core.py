import os
from unittest import TestCase, mock

from interpreter.core.async_core import (
    AsyncInterpreter,
    SENSITIVE_SERVER_SETTINGS,
    Server,
    authenticate_function,
    confirmation_digest,
    ensure_server_api_key,
    is_loopback_host,
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
        with mock.patch.dict(os.environ, {}, clear=True):
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
            clear=True,
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
            clear=True,
        ):
            s = Server(AsyncInterpreter())
            self.assertEqual(s.host, fake_host)
            self.assertEqual(s.port, fake_port)


class TestAsyncServerSecurityHelpers(TestCase):
    def test_is_loopback_host(self):
        self.assertTrue(is_loopback_host("127.0.0.1"))
        self.assertTrue(is_loopback_host("localhost"))
        self.assertTrue(is_loopback_host("::1"))
        self.assertFalse(is_loopback_host("0.0.0.0"))

    def test_confirmation_digest_is_stable(self):
        payload = {"type": "code", "format": "python", "content": "print('hi')"}
        self.assertEqual(
            confirmation_digest(payload),
            confirmation_digest(payload),
        )
        changed = {"type": "code", "format": "python", "content": "print('bye')"}
        self.assertNotEqual(confirmation_digest(payload), confirmation_digest(changed))

    def test_authenticate_function_without_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(authenticate_function(None))
            self.assertTrue(authenticate_function("anything"))

    def test_authenticate_function_with_api_key(self):
        with mock.patch.dict(os.environ, {"INTERPRETER_API_KEY": "secret"}, clear=True):
            self.assertTrue(authenticate_function("secret"))
            self.assertFalse(authenticate_function("wrong"))
            self.assertFalse(authenticate_function(None))

    def test_ensure_server_api_key_loopback_does_not_generate(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ensure_server_api_key("127.0.0.1"))
            self.assertNotIn("INTERPRETER_API_KEY", os.environ)

    def test_ensure_server_api_key_non_loopback_generates(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            generated = ensure_server_api_key("0.0.0.0")
            self.assertIsNotNone(generated)
            self.assertEqual(os.environ["INTERPRETER_API_KEY"], generated)


class TestAsyncApprovalBinding(TestCase):
    def setUp(self):
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False

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


class TestSettingsEndpointGuards(TestCase):
    def test_sensitive_settings_constant_includes_auto_run(self):
        self.assertIn("auto_run", SENSITIVE_SERVER_SETTINGS)

    def test_post_settings_blocks_auto_run(self):
        interpreter = AsyncInterpreter()
        server = Server(interpreter)
        from fastapi.testclient import TestClient

        client = TestClient(server.app)
        response = client.post("/settings", json={"auto_run": True})
        self.assertEqual(response.status_code, 403)
        self.assertIn("auto_run", response.json()["error"])
