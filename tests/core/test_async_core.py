import os
from unittest import TestCase, mock

from interpreter.core.async_core import (
    AsyncInterpreter,
    Server,
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
