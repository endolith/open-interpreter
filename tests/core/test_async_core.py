import json
import os
from unittest import TestCase, mock

from interpreter.core.async_core import (
    AsyncInterpreter,
    Server,
    _format_openai_console_output,
    _is_openai_auxiliary_title_request,
    _normalize_openai_code_approval_reply,
    _new_openai_completion_id,
    _openai_messages_to_lmc,
    _openai_sse_chunk,
)


class TestOpenAICompatHelpers(TestCase):
    def test_console_output_wrapped_in_fence(self):
        out = _format_openai_console_output("line1\nline2", language="bash")
        self.assertIn("```bash", out)
        self.assertIn("line1", out)
        self.assertIsNone(_format_openai_console_output("Note: Shell command output will be shown after completion."))

    def test_code_approval_reply_normalization(self):
        self.assertEqual(_normalize_openai_code_approval_reply("yes"), "yes")
        self.assertEqual(_normalize_openai_code_approval_reply("Yes."), "yes")
        self.assertEqual(_normalize_openai_code_approval_reply("no"), "no")
        self.assertIsNone(_normalize_openai_code_approval_reply("y"))
        self.assertIsNone(_normalize_openai_code_approval_reply("go"))
        self.assertIsNone(_normalize_openai_code_approval_reply("ls"))

    def test_title_request_detection(self):
        prompt = (
            "Based on the chat history, give this conversation a name.\n"
            "Keep it short - 10 words max."
        )
        self.assertTrue(_is_openai_auxiliary_title_request(prompt))
        self.assertFalse(_is_openai_auxiliary_title_request("hi"))

    def test_sse_chunk_id_is_string(self):
        completion_id = _new_openai_completion_id()
        self.assertIsInstance(completion_id, str)
        payload = _openai_sse_chunk(completion_id, 123, delta_content="Hi")
        data_line = payload.strip().removeprefix("data: ")
        chunk = json.loads(data_line)
        self.assertIsInstance(chunk["id"], str)

    def test_openai_system_not_in_lmc_messages(self):
        class _Msg:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        lmc, client_system = _openai_messages_to_lmc(
            [
                _Msg("system", "You are helpful."),
                _Msg("user", "hi"),
            ]
        )
        self.assertEqual(client_system, "You are helpful.")
        self.assertTrue(all(m["role"] != "system" for m in lmc))


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
