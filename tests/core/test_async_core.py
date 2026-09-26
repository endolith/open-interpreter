import hashlib
import json
import os
import threading

import pytest
import socket
from unittest import TestCase, mock

import janus

from interpreter.core.async_core import (
    AsyncInterpreter,
    Server,
    confirmation_digest,
    is_websocket_origin_allowed,
    SENSITIVE_LLM_SETTINGS,
    SENSITIVE_SERVER_SETTINGS,
    authenticate_function,
    complete_message,
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
        """None and empty origins are accepted for non-browser local clients."""
        self.assertTrue(is_websocket_origin_allowed(None))
        self.assertTrue(is_websocket_origin_allowed(""))

    def test_localhost_origins_allowed(self):
        """http origins resolving to localhost are permitted."""
        self.assertTrue(is_websocket_origin_allowed("http://127.0.0.1:8000"))
        self.assertTrue(is_websocket_origin_allowed("http://localhost:8000"))

    def test_remote_origins_rejected(self):
        """Origins outside the local network are rejected."""
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
        """Build a TestClient around a fresh server app, keeping the interpreter."""
        from fastapi.testclient import TestClient

        self.interpreter = AsyncInterpreter()
        self.client = TestClient(Server(self.interpreter).app)

    def _assert_settings_blocked(self, payload, error_substring):
        """POST the given settings payload and assert it is rejected with 403."""
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

    def test_post_settings_applies_a_plain_writable_attribute(self):
        """A non-dict top-level setting is assigned straight onto the interpreter.

        This is the branch a caller reaches for scalar switches; if it stopped
        assigning, POST /settings would report success while changing nothing.
        `context_mode` stands in because auto_run and safe_mode are on the
        sensitive list and are rejected before reaching this branch.
        """
        self.interpreter.context_mode = False

        response = self.client.post("/settings", json={"context_mode": True})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.interpreter.context_mode)

    def test_post_settings_applies_a_nested_llm_attribute(self):
        """A dict setting is applied to the sub-object, not to the interpreter."""
        response = self.client.post("/settings", json={"llm": {"model": "gpt-4o-mini"}})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.interpreter.llm.model, "gpt-4o-mini")

    def test_post_settings_rejects_a_sensitive_key_after_applying_an_earlier_one(self):
        """A sensitive key aborts the payload, leaving earlier keys already applied.

        The loop returns on the first rejected key instead of validating the whole
        payload first, so a multi-key update is applied in part. A caller sending
        a sensitive setting alongside a legitimate one gets a 403 and has to assume
        none of it took effect, which is not what happens.
        """
        sensitive_key = "auto_run"

        response = self.client.post(
            "/settings", json={"context_mode": True, sensitive_key: True}
        )

        self.assertEqual(response.status_code, 403)
        self.assertTrue(self.interpreter.context_mode)

    def test_post_settings_unknown_key_answers_200_with_a_tuple_body(self):
        """An unknown top-level key yields a JSON array and a 200, not a 404.

        The handler returns `({"error": ...}, 404)`, but FastAPI serialises the
        tuple as the response body and answers 200, so the intended status never
        reaches the caller. Pinned as-is: the report asks for characterization
        tests, so this is recorded rather than changed.
        """
        response = self.client.post("/settings", json={"not_a_real_setting": 1})

        self.assertEqual(response.status_code, 200)
        self.assertIn("not found", response.json()[0]["error"])

    def test_post_settings_unknown_llm_subkey_answers_200_with_a_tuple_body(self):
        """An unknown llm sub-key has the same tuple-body-200 shape as above."""
        response = self.client.post("/settings", json={"llm": {"not_a_field": 1}})

        self.assertEqual(response.status_code, 200)
        self.assertIn("not found", response.json()[0]["error"])

    def test_get_setting_unserialisable_value_answers_200_with_a_tuple_body(self):
        """A setting that cannot be JSON-encoded reports the error in a 200 body.

        Same tuple-return shape as the 404 branches: the 500 in
        `({"error": ...}, 500)` never becomes a status code.
        """
        self.interpreter.not_serialisable = object()

        response = self.client.get("/settings/not_serialisable")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Failed to serialize", response.json()[0]["error"])


class TestHomeEndpointTemplate(TestCase):
    """Pins the computed parts of the served chat page.

    The page is assembled at request time from the interpreter's host, port, and
    whether output has to be acknowledged. Until now only the DOCTYPE was
    asserted, so any of those computed fragments could change silently.
    """

    def setUp(self):
        """A TestClient plus the interpreter whose server backs the page."""
        from fastapi.testclient import TestClient

        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.client = TestClient(Server(self.interpreter).app)

    def test_home_is_served_as_html(self):
        """The chat page comes back as text/html."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/html"))

    def test_home_points_the_browser_at_the_host_and_port_actually_serving(self):
        """The embedded websocket URL uses the live server's host and port.

        The template reads host and port per request, so a page generated against
        defaults would point the browser's websocket at the wrong address and the
        chat would silently never connect.
        """
        self.interpreter.server.host = "10.1.2.3"
        self.interpreter.server.port = 4242

        response = self.client.get("/")

        self.assertIn("ws://10.1.2.3:4242/", response.text)

    def test_home_omits_the_ack_handshake_when_acknowledgement_is_off(self):
        """With acknowledgement disabled the page sends no ack frames.

        The ack is what makes the server retry a dropped chunk, so emitting it
        when the server is not waiting for one would leave unsent chunks queued.
        """
        self.interpreter.require_acknowledge = False

        response = self.client.get("/")

        self.assertNotIn('"ack": eventData.id', response.text)

    def test_home_includes_the_ack_handshake_when_acknowledgement_is_on(self):
        """With acknowledgement enabled the page acknowledges each chunk.

        Without this the server would hold every chunk back waiting for an ack
        that never arrives, and the client would show nothing at all.
        """
        self.interpreter.require_acknowledge = True

        response = self.client.get("/")

        self.assertIn('"ack": eventData.id', response.text)

    def test_home_includes_the_approval_controls(self):
        """The page ships the buttons that send go and auth command blocks.

        Without them the only way to approve code would be a raw websocket client,
        so their absence would make manual approval impossible from the browser.
        """
        response = self.client.get("/")

        self.assertIn("approveCodeButton", response.text)
        self.assertIn("authButton", response.text)

    def test_home_sends_go_with_the_last_confirmation_id(self):
        """Approval replays the stored confirmation id, so it cannot go stale.

        go is sent bare only when nothing is pending; the page has to thread the
        id it captured from the confirmation chunk into the command.
        """
        response = self.client.get("/")

        self.assertIn('lastConfirmationId ? ("go:" + lastConfirmationId) : "go"', response.text)


class TestAsyncApprovalBinding(TestCase):
    def setUp(self):
        """An interpreter that pauses at confirmation chunks."""
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False

    def test_confirmation_digest_is_stable(self):
        """The same confirmation payload always produces the same digest."""
        payload = {"type": "code", "format": "python", "content": "print('hi')"}
        self.assertEqual(
            confirmation_digest(payload),
            confirmation_digest(payload),
        )

    def test_approve_pending_confirmation_requires_pending_state(self):
        """Approval is refused when no confirmation is currently pending."""
        self.assertFalse(self.interpreter._approve_pending_confirmation())

    def test_approve_pending_confirmation_accepts_matching_digest(self):
        """Approval succeeds when the provided digest matches the pending one."""
        payload = {"type": "code", "format": "python", "content": "print(1)"}
        self.interpreter.pending_confirmation = payload
        self.interpreter.pending_confirmation_digest = confirmation_digest(payload)

        digest = self.interpreter.pending_confirmation_digest
        self.assertTrue(self.interpreter._approve_pending_confirmation(digest))
        self.assertTrue(self.interpreter._approval_granted)

    def test_approve_pending_confirmation_rejects_wrong_digest(self):
        """Approval is refused when the provided digest does not match."""
        payload = {"type": "code", "format": "python", "content": "print(1)"}
        self.interpreter.pending_confirmation = payload
        self.interpreter.pending_confirmation_digest = confirmation_digest(payload)

        self.assertFalse(self.interpreter._approve_pending_confirmation("deadbeef"))


class TestAsyncInputCommandHandling(TestCase):
    def setUp(self):
        """An interpreter with a live respond thread and queued error output."""
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.interpreter.output_queue = mock.MagicMock()
        self.interpreter.output_queue.sync_q = mock.MagicMock()
        self.interpreter.respond_thread = mock.MagicMock()
        self.interpreter.respond_thread.is_alive.return_value = True

    def _run_go_command(self, command="go"):
        """Feed a complete user command chunk sequence through interpreter.input."""
        import asyncio

        async def run():
            """Feed the start/content/end command chunks to interpreter.input."""
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
        """Return the error chunk contents put onto the sync output queue."""
        return [
            call.args[0].get("content", "")
            for call in self.interpreter.output_queue.sync_q.put.call_args_list
            if call.args[0].get("type") == "error"
        ]

    def test_command_start_does_not_require_join(self):
        """A command start chunk must not join the respond thread."""
        import asyncio

        async def run():
            """Feed a start-only command chunk and check join is never called."""
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


class TestAsyncInputLifecycle(TestCase):
    """Pins the start/content/end dispatch in `input` outside the go/stop commands.

    The command tests elsewhere cover the go paths, but the plain streaming
    lifecycle was only exercised for a command start, which takes the one branch
    that deliberately skips the teardown. A normal start chunk, a content chunk
    and an end chunk all took the other branches untested.
    """

    def setUp(self):
        """An interpreter with a mock output queue and no respond thread."""
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.interpreter.output_queue = mock.MagicMock()
        self.interpreter.output_queue.sync_q = mock.MagicMock()

    def _feed(self, *chunks):
        """Push chunks through interpreter.input on a throwaway event loop."""
        import asyncio

        async def run():
            """Await interpreter.input once per chunk, in order."""
            for chunk in chunks:
                await self.interpreter.input(chunk)

        asyncio.run(run())

    def test_non_command_start_stops_and_joins_a_live_respond_thread(self):
        """A normal start chunk interrupts and drains the running respond thread.

        Starting new work while a response is streaming would interleave two
        responses into one transcript, so the old thread is stopped and joined
        before the new message is accumulated.
        """
        self.interpreter.respond_thread = mock.MagicMock()
        self.interpreter.respond_thread.is_alive.return_value = True

        self._feed({"role": "user", "type": "message", "start": True})

        self.assertTrue(self.interpreter.stop_event.is_set())
        self.interpreter.respond_thread.join.assert_called_once()

    def test_non_command_start_cancels_a_pending_approval(self):
        """Interrupting a paused approval releases the waiting thread as denied.

        The interrupted thread is blocked on the approval event, so joining it
        without cancelling would hang; and leaving the grant set would run the
        code the user was still deciding about.
        """
        self.interpreter.respond_thread = mock.MagicMock()
        self.interpreter.respond_thread.is_alive.return_value = True
        self.interpreter._approval_granted = False

        self._feed({"role": "user", "type": "message", "start": True})

        self.assertTrue(self.interpreter._approval_event.is_set())
        self.assertFalse(self.interpreter._approval_granted)

    def test_non_command_start_without_a_running_thread_never_joins(self):
        """With no respond thread in flight the start chunk only accumulates.

        Joining a thread that was never started would raise, so the liveness
        check has to gate the join rather than merely the stop.
        """
        self.interpreter.respond_thread = None

        self._feed({"role": "user", "type": "message", "start": True})

        self.assertFalse(self.interpreter.stop_event.is_set())
        self.assertEqual(
            self.interpreter.messages, [{"role": "user", "type": "message", "content": ""}]
        )

    def test_content_chunk_only_accumulates(self):
        """A bare content chunk appends and starts no thread and no teardown."""
        self._feed({"role": "user", "type": "message", "start": True})

        self._feed({"type": "message", "content": "hello"})

        self.assertEqual(
            self.interpreter.messages, [{"role": "user", "type": "message", "content": "hello"}]
        )
        self.assertIsNone(self.interpreter.respond_thread)

    def test_end_chunk_starts_a_respond_thread(self):
        """An end chunk on a non-command message launches a fresh respond thread.

        This is the ordinary turn boundary, so a missing start here would leave
        every reply unproduced.
        """
        self.interpreter.respond = mock.MagicMock()
        self._feed({"role": "user", "type": "message", "start": True})

        self._feed({"type": "message", "end": True})

        self.assertIsInstance(self.interpreter.respond_thread, threading.Thread)
        self.interpreter.respond_thread.join(timeout=5)
        self.interpreter.respond.assert_called_once()

    def test_end_chunk_clears_a_leftover_stop_event(self):
        """A new turn clears a stop flag left set by the previous one.

        Without the clear the new respond loop would exit immediately, so the
        reply would never be generated.
        """
        self.interpreter.respond = mock.MagicMock()
        self.interpreter.stop_event.set()
        self._feed({"role": "user", "type": "message", "start": True})

        self._feed({"type": "message", "end": True})

        self.assertFalse(self.interpreter.stop_event.is_set())
        self.interpreter.respond_thread.join(timeout=5)

    def test_end_chunk_discards_the_previous_respond_iterator(self):
        """A new turn drops the iterator held over the previous response.

        Resuming a spent generator would replay or truncate the old response, so
        each turn has to start from a clean iterator.
        """
        self.interpreter.respond = mock.MagicMock()
        self.interpreter._respond_iterator = iter([{"stale": True}])
        self._feed({"role": "user", "type": "message", "start": True})

        self._feed({"type": "message", "end": True})

        self.assertIsNone(self.interpreter._respond_iterator)
        self.interpreter.respond_thread.join(timeout=5)

    def test_stop_command_sets_the_stop_flag_and_joins(self):
        """A stop command halts the in-flight response instead of starting another."""
        self.interpreter.respond_thread = mock.MagicMock()
        self.interpreter.respond_thread.is_alive.return_value = True

        self._feed({"role": "user", "type": "command", "start": True})
        self._feed({"type": "command", "content": "stop"})
        self._feed({"type": "command", "end": True})

        self.assertTrue(self.interpreter.stop_event.is_set())
        self.interpreter.respond_thread.join.assert_called_once()

    def test_stop_command_with_no_thread_still_sets_the_stop_flag(self):
        """Stop is honoured even when nothing is running to be joined."""
        self.interpreter.respond_thread = None

        self._feed({"role": "user", "type": "command", "start": True})
        self._feed({"type": "command", "content": "stop"})
        self._feed({"type": "command", "end": True})

        self.assertTrue(self.interpreter.stop_event.is_set())

    def test_stop_command_cancels_a_pending_approval(self):
        """Stop releases a thread blocked on approval without granting it."""
        self.interpreter.respond_thread = mock.MagicMock()
        self.interpreter.respond_thread.is_alive.return_value = True
        self.interpreter._approval_granted = False

        self._feed({"role": "user", "type": "command", "start": True})
        self._feed({"type": "command", "content": "stop"})
        self._feed({"type": "command", "end": True})

        self.assertTrue(self.interpreter._approval_event.is_set())
        self.assertFalse(self.interpreter._approval_granted)

    def test_command_message_is_removed_from_the_transcript(self):
        """The command chunk is consumed, leaving no "/stop" turn for the model.

        Commands are transport instructions, not conversation, so a leftover
        entry would be replayed to the LLM as something the user said.
        """
        self.interpreter.respond_thread = None

        self._feed({"role": "user", "type": "command", "start": True})
        self._feed({"type": "command", "content": "stop"})
        self._feed({"type": "command", "end": True})

        self.assertEqual(self.interpreter.messages, [])

    def test_failed_approval_stays_silent_without_an_output_queue(self):
        """A refused go emits nothing and raises nothing when no queue exists.

        The queue is optional (an in-process interpreter has none), so the error
        branch has to tolerate its absence instead of dereferencing it.
        """
        self.interpreter.output_queue = None
        self.interpreter.respond_thread = mock.MagicMock()
        self.interpreter.respond_thread.is_alive.return_value = True

        self._feed({"role": "user", "type": "command", "start": True})
        self._feed({"type": "command", "content": "go"})
        self._feed({"type": "command", "end": True})

        self.assertIsNone(self.interpreter.output_queue)
        self.interpreter.respond_thread.start.assert_not_called()


class TestAsyncInterpreterConstruction(TestCase):
    """Pins the environment-driven fields set in AsyncInterpreter.__init__.

    Each of these is read once at construction, so a mutation to a key name, a
    default, or the case-folding would silently misconfigure every later server
    with no error at the point of use.
    """

    def _interpreter_without(self, *names):
        """Build an interpreter with the named env vars guaranteed absent."""
        with mock.patch.dict(os.environ):
            for name in names:
                os.environ.pop(name, None)
            return AsyncInterpreter()

    def test_require_acknowledge_defaults_off(self):
        """Output acknowledgement is off unless the env var asks for it."""
        interpreter = self._interpreter_without("INTERPRETER_REQUIRE_ACKNOWLEDGE")

        self.assertFalse(interpreter.require_acknowledge)

    def test_require_acknowledge_is_on_for_true(self):
        """The exact string "true" enables acknowledgement."""
        with mock.patch.dict(os.environ, {"INTERPRETER_REQUIRE_ACKNOWLEDGE": "true"}):
            self.assertTrue(AsyncInterpreter().require_acknowledge)

    def test_require_acknowledge_is_case_insensitive(self):
        """Any capitalisation of "true" enables it.

        Deployment descriptors vary freely in case, so folding is what keeps the
        setting from being silently ignored.
        """
        for value in ("TRUE", "True", "tRuE"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ, {"INTERPRETER_REQUIRE_ACKNOWLEDGE": value}
                ):
                    self.assertTrue(AsyncInterpreter().require_acknowledge)

    def test_require_acknowledge_stays_off_for_other_values(self):
        """Values that are not "true" leave acknowledgement off.

        Anything truthy-by-accident here would block every response on a prompt
        no client ever sends.
        """
        for value in ("1", "yes", "false", ""):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ, {"INTERPRETER_REQUIRE_ACKNOWLEDGE": value}
                ):
                    self.assertFalse(AsyncInterpreter().require_acknowledge)

    def test_interpreter_id_comes_from_the_environment_when_set(self):
        """INTERPRETER_ID is adopted verbatim so a client can address this server."""
        with mock.patch.dict(os.environ, {"INTERPRETER_ID": "caller-supplied-id"}):
            self.assertEqual(AsyncInterpreter().id, "caller-supplied-id")

    def test_interpreter_id_falls_back_to_a_timestamp(self):
        """With no INTERPRETER_ID the id defaults to the current timestamp.

        The field still has to be unique per process, so the fallback cannot be a
        constant.
        """
        interpreter = self._interpreter_without("INTERPRETER_ID")

        self.assertIsInstance(interpreter.id, float)

    def test_unsent_messages_start_as_an_empty_deque(self):
        """The unsent buffer is an empty deque, ready to append without branching."""
        interpreter = AsyncInterpreter()

        self.assertEqual(list(interpreter.unsent_messages), [])

    def test_interpreter_starts_idle(self):
        """A fresh interpreter has no thread, no queue, and no pending approval.

        Every field that arms later behaviour has to start disarmed, so this
        pins them together as one precondition rather than one test each.
        """
        interpreter = AsyncInterpreter()

        self.assertIsNone(interpreter.respond_thread)
        self.assertIsNone(interpreter.output_queue)
        self.assertIsNone(interpreter._respond_iterator)
        self.assertFalse(interpreter._approval_granted)
        self.assertFalse(interpreter._approval_event.is_set())
        self.assertIsNone(interpreter.pending_confirmation)
        self.assertIsNone(interpreter.pending_confirmation_digest)
        self.assertFalse(interpreter.context_mode)
        self.assertEqual(interpreter.acknowledged_outputs, [])


class TestAsyncRespondApproval(TestCase):
    def setUp(self):
        """An interpreter whose respond output lands on a mock sync queue."""
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.mock_q = mock.MagicMock()
        self.interpreter.output_queue = mock.MagicMock(sync_q=self.mock_q)

    def _confirmation_payload(self):
        """The code payload used by the respond-approval tests."""
        return {"format": "python", "content": "print(1)"}

    def _run_respond_with_chunks(self, chunks, approve=True):
        """Run respond() in a thread over the given chunks, then grant or deny approval."""
        import threading
        import time

        def fake_store():
            """Yield the test chunks as the stored response stream."""
            yield from chunks

        with mock.patch.object(self.interpreter, "_respond_and_store", fake_store):

            def respond_thread():
                """Run respond() to completion on a worker thread."""
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


class TestAsyncRespondProgress(TestCase):
    """Pins the parts of `respond` outside the approval handshake.

    The two existing tests cover granting and denying. Everything else — the
    five-attempt retry when the model returns nothing, the error path, mid-stream
    stops, iterator reuse, and the run_code downgrade — was reachable but
    unasserted, so mutations there survived.
    """

    def setUp(self):
        """An interpreter whose respond output lands on a mock sync queue."""
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.mock_q = mock.MagicMock()
        self.interpreter.output_queue = mock.MagicMock(sync_q=self.mock_q)

    def _respond_over(self, chunks, run_code=None, auto_run=False):
        """Run respond() to completion over a canned chunk stream.

        Returns the interpreter so callers can inspect post-run state; the run
        itself is synchronous because no confirmation chunk is involved.
        """
        self.interpreter.auto_run = auto_run

        def fake_store():
            """Yield the canned chunks as the stored response stream."""
            yield from chunks

        with mock.patch.object(self.interpreter, "_respond_and_store", fake_store):
            self.interpreter.respond(run_code=run_code)
        return self.interpreter

    def _put_chunks(self):
        """Every chunk handed to the output queue during the run."""
        return [call.args[0] for call in self.mock_q.put.call_args_list]

    def test_successful_run_ends_with_the_complete_marker(self):
        """A normal run emits its chunks and then a single complete message.

        The marker is how a client knows the stream ended, so a run that returned
        without it would leave clients waiting forever.
        """
        console = {
            "type": "console",
            "role": "computer",
            "format": "output",
            "content": "done",
        }

        self._respond_over([console])

        self.assertEqual(self._put_chunks(), [console, complete_message])

    def test_run_forwards_chunks_to_the_queue_unchanged(self):
        """Each streamed chunk reaches the queue with its own fields intact."""
        console = {
            "type": "console",
            "role": "computer",
            "format": "output",
            "content": "hello",
        }

        self._respond_over([console])

        self.assertIn(console, self._put_chunks())

    def test_iterator_is_cleared_after_a_completed_run(self):
        """A finished run drops its iterator so the next turn starts fresh.

        Resuming a spent generator would replay the previous answer into the new
        turn, so this has to happen on the normal exit as well as the error one.
        """
        self._respond_over(
            [{"type": "console", "role": "computer", "format": "output", "content": "x"}]
        )

        self.assertIsNone(self.interpreter._respond_iterator)

    def test_an_existing_iterator_is_resumed_not_replaced(self):
        """A pre-set _respond_iterator is consumed as-is on the next run.

        input() deliberately leaves the iterator in place when a turn is
        interrupted mid-approval; recreating it here would discard the chunks
        already produced and restart the model call.
        """
        console = {
            "type": "console",
            "role": "computer",
            "format": "output",
            "content": "resumed",
        }
        self.interpreter._respond_iterator = iter([console])

        def unexpected_store():
            """Fail if respond() asks for a brand new response stream."""
            raise AssertionError("_respond_and_store must not be called")
            yield  # pragma: no cover - makes this a generator

        with mock.patch.object(
            self.interpreter, "_respond_and_store", unexpected_store
        ):
            self.interpreter.respond()

        self.assertIn(console, self._put_chunks())

    def test_first_confirmation_with_run_code_disables_it_and_does_not_pause(self):
        """run_code=True makes the first confirmation execute instead of pausing.

        The user already authorised this turn, so a confirmation arriving under
        that grant must not block waiting for an approval that will never come.
        """
        confirmation = {
            "type": "confirmation",
            "role": "computer",
            "content": {"format": "python", "content": "print(1)"},
        }
        console = {
            "type": "console",
            "role": "computer",
            "format": "output",
            "content": "ran",
        }

        self._respond_over([confirmation, console], run_code=True, auto_run=False)

        put_chunks = self._put_chunks()
        self.assertTrue(any(c.get("content") == "ran" for c in put_chunks))
        self.assertFalse(any(c.get("type") == "error" for c in put_chunks))

    def test_exception_in_the_stream_is_reported_as_an_error_chunk(self):
        """A raising response stream becomes an error chunk, not a propagated error.

        Clients read failures off the queue as a typed message, so an exception
        escaping the loop would leave them with no error and a hung stream.
        """
        def exploding_store():
            """Raise as soon as the stream is pulled."""
            raise ValueError("stream exploded")
            yield  # pragma: no cover - makes this a generator

        with mock.patch.object(self.interpreter, "_respond_and_store", exploding_store):
            self.interpreter.respond()

        errors = [c for c in self._put_chunks() if c.get("type") == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("stream exploded", errors[0]["content"])
        self.assertEqual(errors[0]["role"], "server")

    def test_error_path_also_emits_the_complete_marker(self):
        """Even on failure the stream is closed with a complete message."""
        def exploding_store():
            """Raise as soon as the stream is pulled."""
            raise ValueError("boom")
            yield  # pragma: no cover - makes this a generator

        with mock.patch.object(self.interpreter, "_respond_and_store", exploding_store):
            self.interpreter.respond()

        self.assertEqual(self._put_chunks()[-1], complete_message)

    def test_empty_response_is_retried_five_times_then_raises(self):
        """A model that never answers is nudged five times, then gives up loudly.

        The nudges are what rescue a silent provider; the final raise is what
        stops an infinite loop. Both the bound and the last message matter, so
        they are pinned together.
        """
        with mock.patch.object(self.interpreter, "_respond_and_store", lambda: iter([])):
            with mock.patch("interpreter.core.async_core.time.sleep") as slept:
                with self.assertRaises(Exception) as caught:
                    self.interpreter.respond()

        self.assertIn("No chunks sent", str(caught.exception))
        self.assertEqual(slept.call_count, 5)

    def test_each_retry_appends_a_distinct_nudge(self):
        """Every retry appends a different escalating prompt to the transcript.

        Reusing one message would make retries idempotent and defeat the point, so
        the cycle through the list rather than repeating the first entry.
        """
        with mock.patch.object(self.interpreter, "_respond_and_store", lambda: iter([])):
            with mock.patch("interpreter.core.async_core.time.sleep"):
                with self.assertRaises(Exception):
                    self.interpreter.respond()

        nudges = [m["content"] for m in self.interpreter.messages if m.get("role") == "user"]
        self.assertEqual(len(nudges), 5)
        self.assertEqual(len(set(nudges)), 5)

    def test_final_failure_emits_an_error_chunk_before_raising(self):
        """The give-up path still reports the failure on the queue.

        The raise only reaches the caller; a client attached to the stream learns
        about the failure from this chunk.
        """
        with mock.patch.object(self.interpreter, "_respond_and_store", lambda: iter([])):
            with mock.patch("interpreter.core.async_core.time.sleep"):
                with self.assertRaises(Exception):
                    self.interpreter.respond()

        errors = [c for c in self._put_chunks() if c.get("type") == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("No chunks sent", errors[0]["content"])

    def test_stop_set_while_a_chunk_is_in_flight_halts_before_the_next_put(self):
        """A stop raised between two chunks ends the run without sending the second.

        The stop check runs after a chunk is fetched but before it is forwarded,
        so the in-flight chunk is what gets dropped. Anything already sent stands.
        """
        first = {
            "type": "console",
            "role": "computer",
            "format": "output",
            "content": "first",
        }
        second = {
            "type": "console",
            "role": "computer",
            "format": "output",
            "content": "second",
        }

        def stop_then_second():
            """Emit one chunk, raise the stop flag, then offer a second."""
            yield first
            self.interpreter.stop_event.set()
            yield second

        with mock.patch.object(
            self.interpreter, "_respond_and_store", stop_then_second
        ):
            self.interpreter.respond()

        self.assertEqual(self._put_chunks(), [first])
        self.assertIsNone(self.interpreter._respond_iterator)

    def test_stop_before_the_first_chunk_sends_nothing(self):
        """A stop flag already set means the run produces no output at all.

        This is how a turn interrupted during approval is abandoned; emitting
        anything would leak a response the user already cancelled.
        """
        self.interpreter.stop_event.set()

        self._respond_over(
            [{"type": "console", "role": "computer", "format": "output", "content": "late"}]
        )

        self.assertEqual(self._put_chunks(), [])

    @pytest.mark.timeout(60)
    def test_stop_during_a_wait_cancels_the_pending_approval(self):
        """Stopping while blocked on approval releases the thread as denied.

        The thread is parked on the approval event, so a stop that only set the
        flag would leave it waiting and the code would run once something woke it.
        The stop has to come from another thread, since the response stream is
        suspended at the yield while respond() waits.
        """
        import time

        confirmation = {
            "type": "confirmation",
            "role": "computer",
            "content": {"format": "python", "content": "print(1)"},
        }

        def fake_store():
            """Yield the single confirmation that parks respond()."""
            yield confirmation

        with mock.patch.object(self.interpreter, "_respond_and_store", fake_store):
            worker = threading.Thread(target=self.interpreter.respond)
            worker.start()
            for _ in range(500):
                if self.interpreter.pending_confirmation is not None:
                    break
                time.sleep(0.01)

            self.assertIsNotNone(
                self.interpreter.pending_confirmation,
                "respond() never reached the approval wait",
            )
            self.interpreter.stop_event.set()
            self.interpreter._cancel_pending_approval()
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive())

        self.assertFalse(self.interpreter._approval_granted)
        self.assertIsNone(self.interpreter.pending_confirmation)


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


    def test_run_without_arguments_leaves_host_and_port_untouched(self):
        """run() with no host or port binds to whatever the server already holds.

        The `is not None` guards exist so that omitted values do not overwrite the
        configured ones with None; without them a bare run() would bind to nothing.
        """
        import contextlib
        import io

        s = Server(AsyncInterpreter(), "127.0.0.1", 8123)
        s.uvicorn_server.run = mock.Mock()

        with contextlib.redirect_stdout(io.StringIO()):
            s.run()

        self.assertEqual(s.host, "127.0.0.1")
        self.assertEqual(s.port, 8123)
        s.uvicorn_server.run.assert_called_once_with()

    def test_run_without_zero_host_never_opens_a_socket(self):
        """A loopback or named host prints its address without probing for a LAN IP.

        The probe is a real UDP connect used only to discover the outward-facing
        address. On any host other than 0.0.0.0 the server is not exposed to the
        network, so no socket should be opened at all.
        """
        import contextlib
        import io

        s = Server(AsyncInterpreter(), "127.0.0.1", 8123)
        s.uvicorn_server.run = mock.Mock()

        with mock.patch("interpreter.core.async_core.socket.socket") as socket_cls:
            with contextlib.redirect_stdout(io.StringIO()):
                s.run()

        socket_cls.assert_not_called()

    def test_run_without_zero_host_prints_no_exposure_warning(self):
        """The LAN-exposure warning belongs to the 0.0.0.0 branch only.

        Warning on a loopback bind would cry wolf on the safe, default
        configuration.
        """
        import contextlib
        import io

        s = Server(AsyncInterpreter(), "127.0.0.1", 8123)
        s.uvicorn_server.run = mock.Mock()
        buf = io.StringIO()

        with contextlib.redirect_stdout(buf):
            s.run()

        self.assertNotIn("Warning", buf.getvalue())

    def test_run_closes_the_probe_socket(self):
        """The UDP probe socket is closed after the address is read.

        Leaking it would hold a descriptor for the life of the process, and the
        probe runs on every server start.
        """
        import contextlib
        import io

        s = Server(AsyncInterpreter())
        s.uvicorn_server.run = mock.Mock()
        s.config.host = "0.0.0.0"
        fake_socket = mock.Mock()
        fake_socket.getsockname.return_value = ("192.168.1.50", 12345)

        with mock.patch(
            "interpreter.core.async_core.socket.socket", return_value=fake_socket
        ) as socket_cls:
            with contextlib.redirect_stdout(io.StringIO()):
                s.run()

        socket_cls.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)
        fake_socket.close.assert_called_once()

    def test_run_honours_only_the_host_argument(self):
        """Passing just a host leaves the configured port in place.

        run(host=...) is a valid call, and it must not reset the port to a default.
        The host setter rebuilds the uvicorn server, so the class is patched rather
        than the instance — otherwise the rebuilt server would really start.
        """
        import contextlib
        import io

        s = Server(AsyncInterpreter(), "127.0.0.1", 8123)
        fake_socket = mock.Mock()
        fake_socket.getsockname.return_value = ("192.168.1.50", 12345)

        with mock.patch(
            "interpreter.core.async_core.uvicorn.Server"
        ) as uvicorn_server_cls:
            with mock.patch(
                "interpreter.core.async_core.socket.socket", return_value=fake_socket
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    s.run(host="0.0.0.0")

        self.assertEqual(s.port, 8123)
        self.assertEqual(s.host, "0.0.0.0")
        uvicorn_server_cls.return_value.run.assert_called_once_with()


class TestAuthenticateFunction(TestCase):
    """Pins the API-key check the server's middleware delegates to.

    This decides whether a request is served at all, and its "no key configured"
    case is the one that matters most: get it backwards and a server with no
    configured key either serves everyone or refuses everyone.
    """

    def test_no_configured_key_accepts_any_request(self):
        """With no INTERPRETER_API_KEY set, every request is authorised."""
        with mock.patch.dict(os.environ):
            os.environ.pop("INTERPRETER_API_KEY", None)
            self.assertTrue(authenticate_function(None))
            self.assertTrue(authenticate_function("anything"))

    def test_configured_key_accepts_only_an_exact_match(self):
        """With a key configured, the presented key must equal it."""
        with mock.patch.dict(os.environ, {"INTERPRETER_API_KEY": "supersecret"}):
            self.assertTrue(authenticate_function("supersecret"))
            self.assertFalse(authenticate_function("wrong"))
            self.assertFalse(authenticate_function("SuperSecret"))

    def test_configured_key_rejects_a_missing_header(self):
        """A configured key means an absent header is a failure, not a pass."""
        with mock.patch.dict(os.environ, {"INTERPRETER_API_KEY": "supersecret"}):
            self.assertFalse(authenticate_function(None))

    def test_empty_configured_key_matches_only_an_empty_header(self):
        """An empty INTERPRETER_API_KEY is still a configured key.

        It is a footgun rather than a bug, but the distinction matters: the value
        is compared rather than treated as absent, so only an empty header passes.
        """
        with mock.patch.dict(os.environ, {"INTERPRETER_API_KEY": ""}):
            self.assertTrue(authenticate_function(""))
            self.assertFalse(authenticate_function("supersecret"))


class TestServerAuthenticationWiring(TestCase):
    """Pins how the middleware is attached and where the key check comes from.

    The middleware resolves `self.authenticate` per request, so swapping the
    attribute is the supported way to change the policy; a server that captured
    the function at build time would ignore that.
    """

    def test_server_authenticates_through_the_module_function(self):
        """A freshly built server delegates to authenticate_function."""
        server = Server(AsyncInterpreter())

        self.assertIs(server.authenticate, authenticate_function)

    def test_replacing_authenticate_changes_the_policy(self):
        """Overriding server.authenticate changes what the middleware accepts.

        This is the extension point the attribute exists for; if the middleware
        closed over the module function instead, the override would be inert.
        """
        from fastapi.testclient import TestClient

        server = Server(AsyncInterpreter())
        server.authenticate = lambda key: False
        client = TestClient(server.app)

        response = client.post("/settings", json={"context_mode": True})

        self.assertEqual(response.status_code, 403)

    def test_heartbeat_is_reachable_even_when_auth_rejects_everything(self):
        """/heartbeat stays open so a supervisor can probe a locked server."""
        from fastapi.testclient import TestClient

        server = Server(AsyncInterpreter())
        server.authenticate = lambda key: False
        client = TestClient(server.app)

        response = client.get("/heartbeat")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "alive")

    def test_port_from_the_environment_is_coerced_to_an_integer(self):
        """INTERPRETER_PORT arrives as a string and is stored as an int.

        A string port would reach uvicorn as text and fail to bind, or compare
        unequal to an int port set another way.
        """
        with mock.patch.dict(os.environ, {"INTERPRETER_PORT": "1234"}):
            server = Server(AsyncInterpreter())

        self.assertEqual(server.port, 1234)
        self.assertIsInstance(server.port, int)

    def test_host_and_port_fall_back_to_the_environment(self):
        """With no arguments the env vars supply both, and the defaults cover neither."""
        with mock.patch.dict(
            os.environ, {"INTERPRETER_HOST": "env-host", "INTERPRETER_PORT": "4321"}
        ):
            server = Server(AsyncInterpreter())

        self.assertEqual(server.host, "env-host")
        self.assertEqual(server.port, 4321)

    def test_defaults_apply_when_neither_argument_nor_env_is_set(self):
        """Loopback and 8000 are the defaults, so a bare server is not exposed."""
        with mock.patch.dict(os.environ):
            os.environ.pop("INTERPRETER_HOST", None)
            os.environ.pop("INTERPRETER_PORT", None)
            server = Server(AsyncInterpreter())

        self.assertEqual(server.host, Server.DEFAULT_HOST)
        self.assertEqual(server.port, Server.DEFAULT_PORT)
        self.assertEqual(Server.DEFAULT_HOST, "127.0.0.1")


class TestAsyncOutputQueue(TestCase):
    """Pins the lazily-created output queue that every server reader depends on.

    `output()` is the only producer-facing path onto the queue, and nothing else
    in the class creates it, so the creation branch had no coverage at all:
    mutmut reported both of its mutants as "unmatched to any test". These pin
    both halves of the guard — create when absent, reuse when present.
    """

    def setUp(self):
        """A fresh interpreter, whose output_queue starts as None."""
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False

    def test_output_queue_starts_absent(self):
        """The constructor leaves output_queue unset; only output() creates it."""
        self.assertIsNone(self.interpreter.output_queue)

    def test_output_creates_the_queue_when_absent(self):
        """Calling output() on a fresh interpreter creates the janus queue.

        Without this the lazy init would never run, so a client reading output
        would await a queue that does not exist. The task is cancelled once the
        creation is observed, since output() then blocks on an empty queue.
        """
        import asyncio

        async def run():
            """Start output(), let it reach its await, then inspect the queue."""
            task = asyncio.ensure_future(self.interpreter.output())
            await asyncio.sleep(0)
            try:
                self.assertIsNotNone(self.interpreter.output_queue)
            finally:
                task.cancel()

        asyncio.run(run())

    def test_output_reuses_an_existing_queue(self):
        """An already-created queue is reused, not replaced by a fresh empty one.

        Replacing it would drop any messages already buffered for the client and
        strand the writer on the orphaned queue, so identity has to hold.
        """
        import asyncio

        async def run():
            """Pre-load a queue, then confirm output() drains that same queue."""
            queue = janus.Queue()
            await queue.async_q.put({"role": "assistant", "content": "buffered"})
            self.interpreter.output_queue = queue

            received = await self.interpreter.output()

            self.assertEqual(received, {"role": "assistant", "content": "buffered"})
            self.assertIs(self.interpreter.output_queue, queue)

        asyncio.run(run())


class TestConfirmationDigestContent(TestCase):
    """Pins exactly what confirmation_digest hashes, not just that it is stable.

    The existing test only asserted digest(payload) == digest(payload), which any
    function returning a constant would satisfy. These pin the actual preimage so
    a mutation to the keys or defaults changes the digest.
    """

    def test_digest_hashes_format_and_content_with_nul_separator(self):
        """The digest is sha256 over "<format>\\0<content>".

        Pinning the preimage ties the digest to both fields and to the separator,
        so changing either the key read or the default produces a different hash.
        Approval is bound to the exact code the user reviewed, so a digest that
        ignored the language or collided across separators would mis-bind it.
        """
        payload = {"type": "code", "format": "python", "content": "print('hi')"}
        expected = hashlib.sha256(b"python\x00print('hi')").hexdigest()

        self.assertEqual(confirmation_digest(payload), expected)

    def test_digest_defaults_absent_fields_to_empty_string(self):
        """A payload missing format or content substitutes "", not None.

        The separator still has to appear, so a payload with no fields hashes the
        two NUL-separated empties. A None default would hash the literal "None"
        and make two differently-absent payloads collide.
        """
        expected = hashlib.sha256(b"\x00").hexdigest()

        self.assertEqual(confirmation_digest({}), expected)
        self.assertEqual(confirmation_digest({"format": "python"}), hashlib.sha256(b"python\x00").hexdigest())
        self.assertEqual(confirmation_digest({"content": "x = 1"}), hashlib.sha256(b"\x00x = 1").hexdigest())

    def test_digest_changes_when_only_the_format_changes(self):
        """Same code in a different language must not share a digest.

        Without this, a digest that ignored the language would let a digest
        issued for one language approve a confirmation in another.
        """
        python_digest = confirmation_digest({"format": "python", "content": "print(1)"})
        js_digest = confirmation_digest({"format": "javascript", "content": "print(1)"})

        self.assertNotEqual(python_digest, js_digest)

    def test_digest_changes_when_only_the_content_changes(self):
        """Identical language with different code must produce a different digest."""
        first = confirmation_digest({"format": "python", "content": "print(1)"})
        second = confirmation_digest({"format": "python", "content": "print(2)"})

        self.assertNotEqual(first, second)

    def test_digest_ignores_keys_other_than_format_and_content(self):
        """Only format and content feed the digest; other keys cannot move it.

        Approvals are looked up by digest while the pending payload carries a
        type, so if an extra key leaked into the preimage the stored digest and
        a recomputed one would disagree and every approval would be refused.
        """
        without_type = confirmation_digest({"format": "python", "content": "x = 1"})
        with_type = confirmation_digest(
            {"type": "code", "format": "python", "content": "x = 1", "role": "assistant"}
        )

        self.assertEqual(without_type, with_type)


class TestAsyncPendingStateResets(TestCase):
    """Pins the two teardown helpers that unblock or cancel a paused respond thread.

    Both are called on paths that keep a thread waiting, so a no-op reset would
    leave the interpreter permanently stuck awaiting approval that can no longer
    be granted.
    """

    def setUp(self):
        """An interpreter with a confirmation pending and approval ungranted."""
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False

    def test_clear_pending_confirmation_clears_both_fields(self):
        """Clearing drops the payload and its digest together.

        Leaving the digest behind would let a later approval match a confirmation
        whose code is no longer the one on screen.
        """
        self.interpreter.pending_confirmation = {"format": "python", "content": "x = 1"}
        self.interpreter.pending_confirmation_digest = "abc123"

        self.interpreter._clear_pending_confirmation()

        self.assertIsNone(self.interpreter.pending_confirmation)
        self.assertIsNone(self.interpreter.pending_confirmation_digest)

    def test_clear_pending_confirmation_makes_fresh_approval_impossible(self):
        """After a reset, a previously valid digest no longer approves anything."""
        payload = {"format": "python", "content": "x = 1"}
        digest = confirmation_digest(payload)
        self.interpreter.pending_confirmation = payload
        self.interpreter.pending_confirmation_digest = digest

        self.interpreter._clear_pending_confirmation()

        self.assertFalse(self.interpreter._approve_pending_confirmation(digest))

    def test_cancel_pending_approval_unblocks_and_denies(self):
        """Cancelling releases the waiting thread but records the request as denied.

        The event must be set or the thread waits forever; the flag must stay
        False or a cancelled approval would be indistinguishable from a granted
        one and the paused code would run.
        """
        self.assertFalse(self.interpreter._approval_granted)
        self.assertFalse(self.interpreter._approval_event.is_set())

        self.interpreter._cancel_pending_approval()

        self.assertTrue(self.interpreter._approval_event.is_set())
        self.assertFalse(self.interpreter._approval_granted)

    def test_cancel_pending_approval_overrides_a_previously_granted_approval(self):
        """A cancel after a grant still ends up denied, not approved.

        The interrupt path sets the flag and cancels in sequence, so the cancel
        has to be the last word for the code not to run.
        """
        payload = {"format": "python", "content": "x = 1"}
        self.interpreter.pending_confirmation = payload
        self.interpreter.pending_confirmation_digest = confirmation_digest(payload)
        self.assertTrue(self.interpreter._approve_pending_confirmation())

        self.interpreter._cancel_pending_approval()

        self.assertFalse(self.interpreter._approval_granted)

    def test_reset_respond_iterator_clears_pending_confirmation(self):
        """Resetting the iterator also drops the pending confirmation.

        Each new turn must need fresh approval, so a digest carried over from the
        previous turn would let a new confirmation inherit an old approval.
        """
        self.interpreter._respond_iterator = iter([])
        self.interpreter.pending_confirmation = {"format": "python", "content": "x = 1"}
        self.interpreter.pending_confirmation_digest = "abc123"

        self.interpreter._reset_respond_iterator()

        self.assertIsNone(self.interpreter._respond_iterator)
        self.assertIsNone(self.interpreter.pending_confirmation)


class TestAsyncAccumulate(TestCase):
    """Pins how streamed LMC chunks become messages.

    `accumulate` decides, per chunk, whether to extend the current message, start
    a new one, ignore the chunk, or refuse it. Every one of those branches was
    reachable by the existing tests but none of them asserted the resulting
    `messages`, so a mutation to any condition went unnoticed. These assert the
    message list after each chunk shape.
    """

    def setUp(self):
        """An interpreter with no messages accumulated yet."""
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.interpreter.messages = []

    def test_json_string_chunk_is_parsed_and_starts_a_message(self):
        """A chunk arriving as a JSON string is parsed before anything else looks at it.

        The websocket path delivers strings, so without the parse the chunk is not
        a dict and no branch would fire at all. Sending a start chunk then a
        content chunk, both as strings, shows the parse feeds the normal path.
        """
        self.interpreter.accumulate(
            json.dumps({"role": "user", "type": "message", "start": True})
        )
        self.interpreter.accumulate(json.dumps({"type": "message", "content": "hi"}))

        self.assertEqual(
            self.interpreter.messages, [{"role": "user", "type": "message", "content": "hi"}]
        )

    def test_start_and_content_in_one_chunk_takes_the_content_path(self):
        """A chunk carrying both flags is treated as content, not as a start.

        "content" is tested before "start", so on an empty transcript such a chunk
        is refused rather than opening a message. This surprises callers, and the
        ordering is what makes the refusal happen, so it is pinned deliberately.
        """
        with self.assertRaises(Exception) as caught:
            self.interpreter.accumulate(
                {"role": "user", "type": "message", "start": True, "content": "hi"}
            )

        self.assertIn("start: True", str(caught.exception))
        self.assertEqual(self.interpreter.messages, [])

    def test_active_line_chunks_are_ignored(self):
        """A chunk formatted "active_line" is dropped without touching messages.

        These stream the current shell line continuously; appending them would
        corrupt the transcript with terminal echo.
        """
        self.interpreter.messages = [{"role": "user", "type": "message", "content": "kept"}]

        self.interpreter.accumulate({"role": "output", "format": "active_line", "content": "ls -la"})

        self.assertEqual(
            self.interpreter.messages, [{"role": "user", "type": "message", "content": "kept"}]
        )

    def test_content_before_any_message_is_refused(self):
        """Content arriving with no message to attach to raises rather than guessing.

        Creating an implicit message would let a client that skipped the start
        chunk inject content the user never began.
        """
        with self.assertRaises(Exception) as caught:
            self.interpreter.accumulate({"role": "user", "type": "message", "content": "orphan"})

        self.assertIn("start: True", str(caught.exception))
        self.assertEqual(self.interpreter.messages, [])

    def test_matching_type_and_format_extends_the_current_message(self):
        """A continuation chunk with the same type and format appends its content."""
        self.interpreter.messages = [
            {"role": "user", "type": "message", "format": "output", "content": "print("}
        ]

        self.interpreter.accumulate(
            {"type": "message", "format": "output", "content": "1)"}
        )

        self.assertEqual(
            self.interpreter.messages,
            [{"role": "user", "type": "message", "format": "output", "content": "print(1)"}],
        )

    def test_differing_type_starts_a_new_message(self):
        """A chunk whose type differs begins a new message instead of appending.

        Code and message chunks share the stream, so appending across a type
        change would fuse generated code into the user's sentence.
        """
        self.interpreter.messages = [
            {"role": "user", "type": "message", "format": "output", "content": "run this"}
        ]

        self.interpreter.accumulate({"type": "code", "format": "python", "content": "print(1)"})

        self.assertEqual(
            self.interpreter.messages,
            [
                {"role": "user", "type": "message", "format": "output", "content": "run this"},
                {"type": "code", "format": "python", "content": "print(1)"},
            ],
        )

    def test_differing_format_alone_starts_a_new_message(self):
        """A chunk whose format differs but type matches still starts a new message.

        Switching output formats mid-message would blend two streams together, so
        format has to break the message even when the type agrees.
        """
        self.interpreter.messages = [
            {"role": "user", "type": "message", "format": "output", "content": "before"}
        ]

        self.interpreter.accumulate({"type": "message", "format": "code", "content": "after"})

        self.assertEqual(len(self.interpreter.messages), 2)
        self.assertEqual(self.interpreter.messages[1]["content"], "after")

    def test_start_chunk_drops_the_start_flag_and_defaults_content(self):
        """A start-only chunk creates a message with content defaulted to "".

        The "start" flag is protocol bookkeeping and must not reach the model,
        and content has to exist so later chunks can append to it.
        """
        self.interpreter.accumulate({"role": "user", "type": "message", "start": True})

        self.assertEqual(
            self.interpreter.messages, [{"role": "user", "type": "message", "content": ""}]
        )
        self.assertNotIn("start", self.interpreter.messages[0])

    def test_start_chunk_does_not_mutate_the_callers_dict(self):
        """Accumulating pops "start" from a copy, leaving the caller's chunk intact.

        The caller reuses its chunk dict across the websocket, so popping in place
        would make the second delivery lose its start flag.
        """
        chunk = {"role": "user", "type": "message", "start": True}

        self.interpreter.accumulate(chunk)

        self.assertIn("start", chunk)

    def test_type_is_backfilled_onto_a_type_less_message(self):
        """A continuation backfills "type" when the start chunk omitted it.

        A type-less start is legal, but leaving it type-less would make every
        later chunk look like a type change and fork a message per token.
        """
        self.interpreter.messages = [{"role": "user", "content": ""}]

        self.interpreter.accumulate({"type": "message", "content": "hi"})

        self.assertEqual(
            self.interpreter.messages, [{"role": "user", "type": "message", "content": "hi"}]
        )

    def test_format_is_backfilled_onto_a_format_less_message(self):
        """A continuation backfills "format" when the start chunk omitted it.

        Without the backfill the message would look format-less forever and every
        subsequent chunk would fork a new message.
        """
        self.interpreter.messages = [{"role": "user", "type": "message", "content": ""}]

        self.interpreter.accumulate({"type": "message", "format": "output", "content": "hi"})

        self.assertEqual(
            self.interpreter.messages,
            [{"role": "user", "type": "message", "format": "output", "content": "hi"}],
        )

    def test_bytes_chunk_converts_an_empty_content_placeholder_first(self):
        """A bytes chunk re-initialises an empty "" placeholder as b"" before adding.

        Adding bytes to the empty string would raise, so the placeholder has to be
        converted to bytes first or binary output could never be accumulated.
        """
        self.interpreter.messages = [{"role": "user", "type": "message", "content": ""}]

        self.interpreter.accumulate(b"\x00\x01binary")

        self.assertEqual(
            self.interpreter.messages[0]["content"], b"\x00\x01binary"
        )

    def test_bytes_chunk_appends_to_existing_bytes_content(self):
        """A bytes chunk appends to content that is already bytes."""
        self.interpreter.messages = [{"role": "user", "type": "message", "content": b"\x00"}]

        self.interpreter.accumulate(b"\x01")

        self.assertEqual(self.interpreter.messages[0]["content"], b"\x00\x01")

    def test_unsupported_chunk_type_is_ignored(self):
        """A chunk that is neither dict, str, nor bytes leaves messages untouched."""
        self.interpreter.messages = [{"role": "user", "type": "message", "content": "kept"}]

        self.interpreter.accumulate(42)

        self.assertEqual(
            self.interpreter.messages, [{"role": "user", "type": "message", "content": "kept"}]
        )

    def test_dict_without_start_or_content_is_ignored(self):
        """A dict carrying neither "start" nor "content" adds nothing.

        Metadata-only chunks must not spawn empty messages that would then be sent
        to the model as blank turns.
        """
        self.interpreter.messages = [{"role": "user", "type": "message", "content": "kept"}]

        self.interpreter.accumulate({"role": "assistant", "type": "message"})

        self.assertEqual(len(self.interpreter.messages), 1)


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
