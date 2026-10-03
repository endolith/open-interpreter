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


def wait_until(predicate, timeout_message, *, fail, timeout=5):
    """Block until predicate() is truthy, or fail with timeout_message.

    Waits on a threading.Event rather than time.sleep because the approval tests
    patch interpreter.core.async_core.time.sleep, which is the shared time
    module: a time.sleep poll returns instantly and burns all its iterations
    before the worker thread is ever scheduled.

    Exceeding the timeout calls fail() rather than returning, so a caller cannot
    fall through to an assertion that reports a confusing None mismatch instead
    of the real problem.
    """
    poll = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        poll.wait(0.01)
    fail(timeout_message)


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


class _UnauthenticatedServerTestCase(TestCase):
    """Base that hides INTERPRETER_API_KEY from the auth middleware for one test.

    The middleware resolves `server.authenticate` on every request, and
    `authenticate_function` re-reads the environment each time. So an exported
    INTERPRETER_API_KEY turns every request these tests make into a 403, and they
    would fail for a reason unrelated to what they check. The websocket tests
    already do this via a `monkeypatch.delenv` fixture; unittest classes need the
    patch started and stopped by hand.
    """

    def setUp(self):
        """Clear the key for this test, restoring the environment afterwards."""
        env_patch = mock.patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        os.environ.pop("INTERPRETER_API_KEY", None)
        super().setUp()


class TestSettingsEndpointGuards(_UnauthenticatedServerTestCase):
    def setUp(self):
        """Build a TestClient around a fresh server app, keeping the interpreter."""
        from fastapi.testclient import TestClient

        super().setUp()
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


class TestHomeEndpointTemplate(_UnauthenticatedServerTestCase):
    """Pins the computed parts of the served chat page.

    The page is assembled at request time from the interpreter's host, port, and
    whether output has to be acknowledged. Until now only the DOCTYPE was
    asserted, so any of those computed fragments could change silently.
    """

    def setUp(self):
        """A TestClient plus the interpreter whose server backs the page."""
        from fastapi.testclient import TestClient

        super().setUp()
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

        # Bounded: a regression in the run_code grant would park on the approval
        # wait instead of running, so an unbounded call would hang the suite.
        worker = threading.Thread(
            target=self._respond_over,
            args=([confirmation, console],),
            kwargs={"run_code": True, "auto_run": False},
            daemon=True,
        )
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())

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
        # The prompts are messages; a missing or mistyped type would make the
        # provider ignore them and retry a silent turn forever.
        nudge_dicts = [m for m in self.interpreter.messages if m.get("role") == "user"]
        self.assertTrue(all(m["type"] == "message" for m in nudge_dicts))

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
            worker = threading.Thread(target=self.interpreter.respond, daemon=True)
            worker.start()
            for _ in range(500):
                # Wait for complete_message, not pending_confirmation. respond()
                # sets pending_confirmation early, then clears the approval event
                # and only afterwards puts complete_message before waiting. A
                # cancel sent in between would be erased by that clear(), leaving
                # the worker blocked in wait() forever — and since the thread is
                # not the main one, pytest would hang at shutdown rather than
                # report the failure.
                if complete_message in self._put_chunks():
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

    def test_chunks_with_nothing_to_accumulate_are_ignored(self):
        """Carriage and unsupported chunk shapes leave the transcript untouched.

        Three shapes reach no branch: an active_line update (the shell echoes the
        current line continuously, and appending it would corrupt the transcript),
        a non-dict/non-str/non-bytes value, and a dict carrying neither "start"
        nor "content". Each would otherwise spawn an empty message that then
        reaches the model as a blank turn.
        """
        kept = [{"role": "user", "type": "message", "content": "kept"}]
        ignored_chunks = {
            "active_line update": {
                "role": "output",
                "type": "console",
                "format": "active_line",
                "content": "ls -la",
            },
            "unsupported type": 42,
            "no start or content": {"role": "assistant", "type": "message"},
        }

        for label, chunk in ignored_chunks.items():
            with self.subTest(chunk=label):
                self.interpreter.messages = list(kept)

                self.interpreter.accumulate(chunk)

                self.assertEqual(self.interpreter.messages, kept)

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


class TestAsyncInterpreterArgumentForwarding(TestCase):
    """Pins that AsyncInterpreter.__init__ forwards arguments and exact defaults.

    Mutation testing showed these lines survive untouched: dropping *args or
    **kwargs, None-for-False swaps on the disarmed flags, and building the
    embedded server for a different interpreter all pass the existing suite,
    because nothing asserted the wiring itself rather than downstream effects.
    """

    def test_positional_arguments_reach_the_base_class(self):
        """A transcript passed positionally must land on the base interpreter."""
        messages = [{"role": "user", "type": "message", "content": "hi"}]

        self.assertEqual(AsyncInterpreter(messages).messages, messages)

    def test_keyword_arguments_reach_the_base_class(self):
        """Keyword settings such as auto_run must reach the base interpreter."""
        self.assertTrue(AsyncInterpreter(auto_run=True).auto_run)

    def test_initial_flag_values_are_exactly_false(self):
        """Disarmed flags start as False itself, not merely a falsy value.

        None would behave the same everywhere these flags are read today, so
        only an identity check pins the documented initial state against
        None-for-False swaps.
        """
        interpreter = AsyncInterpreter()

        self.assertIs(interpreter._approval_granted, False)
        self.assertIs(interpreter.print, False)
        self.assertIs(interpreter.context_mode, False)

    def test_server_is_built_for_this_interpreter(self):
        """The embedded server must close over this interpreter, not None."""
        with mock.patch("interpreter.core.async_core.Server") as server_cls:
            interpreter = AsyncInterpreter()

        server_cls.assert_called_once_with(interpreter)
        self.assertIs(interpreter.server, server_cls.return_value)


class TestServerConfigIdentity(TestCase):
    """Pins that the uvicorn config serves this app and stays the live object."""

    def test_config_serves_this_apps_routes(self):
        """uvicorn.Config must wrap this server's FastAPI app, not None."""
        server = Server(AsyncInterpreter())

        self.assertIs(server.config.app, server.app)

    def test_uvicorn_server_wraps_the_live_config(self):
        """The running server must wrap the config object the properties expose."""
        server = Server(AsyncInterpreter())

        self.assertIs(server.uvicorn_server.config, server.config)


class TestAsyncRespondApprovalValues(TestCase):
    """Pins the values carried through the approval handshake.

    The existing approval tests prove the handshake pauses and resumes; they
    never assert what is recorded while it waits (the pending payload, its
    digest, the reset grant flag) or what a denial leaves behind, so mutations
    to those values survived.
    """

    def setUp(self):
        """An interpreter whose respond output lands on a mock sync queue."""
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.mock_q = mock.MagicMock()
        self.interpreter.output_queue = mock.MagicMock(sync_q=self.mock_q)

    def _confirmation_payload(self):
        """The code payload used by the approval-value tests."""
        return {"format": "python", "content": "print(1)"}

    def _wait_for_pending_confirmation(self, timeout=5):
        """Block until respond() parks on the approval wait, then return."""
        wait_until(
            lambda: self.interpreter.pending_confirmation is not None,
            "respond() never reached the approval wait",
            fail=self.fail,
            timeout=timeout,
        )

    def test_approval_wait_exposes_the_payload_digest_and_a_denied_grant(self):
        """While parked, the payload, its digest, and a False grant are visible.

        A client approving over the socket reads exactly these fields, so a
        mutation that records the wrong payload, skips the digest, or leaves a
        stale grant would approve the wrong code or skip the wait silently.
        """
        confirmation = {
            "type": "confirmation",
            "role": "computer",
            "content": self._confirmation_payload(),
        }

        def fake_store():
            """Yield the single confirmation that parks respond()."""
            yield confirmation

        with mock.patch.object(self.interpreter, "_respond_and_store", fake_store):
            worker = threading.Thread(target=self.interpreter.respond, daemon=True)
            worker.start()
            self._wait_for_pending_confirmation()

            self.assertEqual(
                self.interpreter.pending_confirmation, self._confirmation_payload()
            )
            self.assertEqual(
                self.interpreter.pending_confirmation_digest,
                confirmation_digest(self._confirmation_payload()),
            )
            self.assertIs(self.interpreter._approval_granted, False)

            self.assertTrue(self.interpreter._approve_pending_confirmation())
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive())

        self.assertIsNone(self.interpreter.pending_confirmation)
        self.assertIsNone(self.interpreter.pending_confirmation_digest)

    def test_denied_approval_closes_the_stream_with_two_complete_markers(self):
        """A denial emits the pre-wait marker plus the denial marker, then stops.

        The first marker was already sent before the wait began; the second one
        is what tells the client the turn ended without running anything.
        """
        confirmation = {
            "type": "confirmation",
            "role": "computer",
            "content": self._confirmation_payload(),
        }

        def fake_store():
            """Yield the single confirmation that parks respond()."""
            yield confirmation

        with mock.patch.object(self.interpreter, "_respond_and_store", fake_store):
            worker = threading.Thread(target=self.interpreter.respond, daemon=True)
            worker.start()
            self._wait_for_pending_confirmation()

            self.interpreter._approval_granted = False
            self.interpreter._approval_event.set()
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive())

        put_chunks = [call.args[0] for call in self.mock_q.put.call_args_list]
        self.assertEqual(put_chunks[-2:], [complete_message, complete_message])
        self.assertIsNone(self.interpreter._respond_iterator)

    def test_approval_of_a_lone_confirmation_completes_without_retry(self):
        """An approved turn with no trailing chunks is done: sent, not retried.

        The confirmation chunk itself sets the sent flag, so a run that ends
        right after approval must close instead of nudging an empty model.
        """
        confirmation = {
            "type": "confirmation",
            "role": "computer",
            "content": self._confirmation_payload(),
        }

        def fake_store():
            """Yield only the confirmation, then go silent."""
            yield confirmation

        with mock.patch.object(self.interpreter, "_respond_and_store", fake_store):
            with mock.patch("interpreter.core.async_core.time.sleep"):
                worker = threading.Thread(
                    target=self.interpreter.respond, daemon=True
                )
                worker.start()
                self._wait_for_pending_confirmation()

                self.assertTrue(self.interpreter._approve_pending_confirmation())
                worker.join(timeout=10)
                self.assertFalse(worker.is_alive())

        nudges = [
            m
            for m in self.interpreter.messages
            if m.get("role") == "user"
        ]
        self.assertEqual(nudges, [])
        self.assertIn(complete_message, [call.args[0] for call in self.mock_q.put.call_args_list])


class TestAsyncRespondDisplay(TestCase):
    """Pins the terminal rendering of streamed chunks when print is on.

    With self.print False (the default) this whole block is dead code, so every
    branch — code fences, message text, the active-line skip, the image notice,
    and the ascii pipeline — survived mutation testing unasserted.
    """

    def setUp(self):
        """An interpreter that prints instead of staying silent, queue mocked."""
        self.interpreter = AsyncInterpreter()
        self.interpreter.print = True
        self.interpreter.auto_run = False
        self.mock_q = mock.MagicMock()
        self.interpreter.output_queue = mock.MagicMock(sync_q=self.mock_q)

    def _printed_output_of(self, chunks):
        """Run respond() over the chunks and return everything printed."""
        import contextlib
        import io

        def fake_store():
            """Yield the canned chunks as the stored response stream."""
            yield from chunks

        buffer = io.StringIO()
        with mock.patch.object(self.interpreter, "_respond_and_store", fake_store):
            with contextlib.redirect_stdout(buffer):
                self.interpreter.respond()
        return buffer.getvalue()

    def test_code_start_prints_an_opening_fence(self):
        """A code start chunk opens a fenced block naming the language."""
        chunk = {
            "type": "code",
            "role": "computer",
            "format": "python",
            "start": True,
        }

        self.assertIn("```python", self._printed_output_of([chunk]))

    def test_code_end_prints_a_closing_fence(self):
        """A code end chunk closes the fenced block it opened."""
        chunk = {
            "type": "code",
            "role": "computer",
            "format": "python",
            "end": True,
        }

        self.assertIn("```", self._printed_output_of([chunk]))

    def test_message_content_is_printed_verbatim(self):
        """A plain message chunk reaches the terminal unchanged."""
        chunk = {"type": "message", "role": "assistant", "content": "hello there"}

        self.assertIn("hello there", self._printed_output_of([chunk]))

    def test_non_ascii_content_is_stripped_to_ascii(self):
        """The ascii pipeline drops characters the terminal cannot show."""
        chunk = {
            "type": "console",
            "role": "computer",
            "format": "output",
            "content": "h\u00e9llo w\u00f6rld",
        }

        printed = self._printed_output_of([chunk])

        self.assertIn("hllo wrld", printed)
        self.assertNotIn("\u00e9", printed)

    def test_active_line_chunks_are_not_printed(self):
        """Live execution line markers never reach the terminal text."""
        chunk = {
            "type": "console",
            "role": "computer",
            "format": "active_line",
            "content": "line 5",
        }

        self.assertEqual(self._printed_output_of([chunk]), "")

    def test_image_chunks_print_a_placeholder_instead_of_bytes(self):
        """Base64 image data is replaced by a short notice, never dumped raw."""
        chunk = {
            "type": "console",
            "role": "computer",
            "format": "base64.png",
            "content": "iVBORw0KGgo=",
        }

        printed = self._printed_output_of([chunk])

        self.assertIn("[An image was produced]", printed)
        self.assertNotIn("iVBORw0KGgo=", printed)

    def test_debug_mode_logs_each_chunk(self):
        """With debug on, every chunk is echoed to the log line by line."""
        import contextlib
        import io

        chunk = {
            "type": "console",
            "role": "computer",
            "format": "output",
            "content": "debugged",
        }
        self.interpreter.debug = True

        def fake_store():
            """Yield the single chunk the debug log must echo."""
            yield chunk

        buffer = io.StringIO()
        with mock.patch.object(self.interpreter, "_respond_and_store", fake_store):
            with contextlib.redirect_stdout(buffer):
                self.interpreter.respond()

        self.assertIn("produced this chunk", buffer.getvalue())


class TestOpenAIGeneratorRunCodePath(TestCase):
    """Pins openai_compatible_generator with run_code=True (the "yes" turn).

    When the user confirms code execution, the generator streams the stored
    turn verbatim as SSE frames: messages as content frames, code as fenced
    frames, anything else as nothing. Mutation testing showed all three
    branches survive because no test posts a "yes" turn and reads the frames.
    """

    def setUp(self):
        """A TestClient with auth open, pointed at a fresh interpreter."""
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("INTERPRETER_API_KEY", None)
        from fastapi.testclient import TestClient

        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.client = TestClient(Server(self.interpreter).app)

    def _sse_payloads(self, text):
        """Parse the data: lines of an SSE body back into payload dicts."""
        payloads = []
        for line in text.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[len("data: "):]))
        return payloads

    def _post_yes_turn(self, chunks):
        """POST a "yes" reply while the stored turn yields the chunks."""
        self.interpreter.messages = [
            {
                "role": "assistant",
                "type": "code",
                "format": "python",
                "content": "print(1)",
            }
        ]

        def fake_store():
            """Yield the canned stored-turn chunks."""
            yield from chunks

        with mock.patch.object(
            self.interpreter, "_respond_and_store", fake_store
        ):
            response = self.client.post(
                "/openai/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "yes"}],
                    "stream": True,
                },
            )
        self.assertEqual(response.status_code, 200)
        return self._sse_payloads(response.text)

    def test_message_chunks_become_content_frames_with_envelope(self):
        """A message chunk becomes one frame carrying the chunk envelope."""
        payloads = self._post_yes_turn(
            [{"type": "message", "role": "assistant", "content": "hello"}]
        )

        self.assertEqual(len(payloads), 1)
        frame = payloads[0]
        self.assertEqual(frame["object"], "chat.completion.chunk")
        self.assertEqual(frame["model"], "open-interpreter")
        self.assertEqual(frame["choices"][0]["delta"]["content"], "hello")
        self.assertEqual(frame["id"], 0)

    def test_code_chunks_become_fenced_frames_in_order(self):
        """A code turn streams as fence-open, body, fence-close frames."""
        chunks = [
            {"type": "code", "role": "computer", "format": "python", "start": True},
            {
                "type": "code",
                "role": "computer",
                "format": "python",
                "content": "x = 1",
            },
            {"type": "code", "role": "computer", "format": "python", "end": True},
        ]

        payloads = self._post_yes_turn(chunks)

        self.assertEqual(
            [p["choices"][0]["delta"]["content"] for p in payloads],
            ["```python\n", "x = 1", "\n```\n"],
        )
        self.assertEqual([p["id"] for p in payloads], [0, 1, 2])

    def test_chunks_without_displayable_content_emit_no_frames(self):
        """Status and progress chunks pass through silently on the stream."""
        payloads = self._post_yes_turn(
            [{"type": "status", "role": "server", "content": "working"}]
        )

        self.assertEqual(payloads, [])


class TestOpenAIGeneratorChatPath(TestCase):
    """Pins openai_compatible_generator with run_code False/None (chat turns).

    A plain chat message is retried across the fixed prompt list until chunks
    arrive: confirmations surface as an approval prompt, silent prompts are
    retried, and the stop flag suppresses everything. None of this was posted
    through the endpoint before, so the whole branch survived.
    """

    def setUp(self):
        """A TestClient with auth open, pointed at a fresh interpreter."""
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("INTERPRETER_API_KEY", None)
        from fastapi.testclient import TestClient

        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.client = TestClient(Server(self.interpreter).app)

    def _post_chat(self):
        """POST a plain chat message and return the parsed SSE payloads."""
        response = self.client.post(
            "/openai/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payloads = []
        for line in response.text.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[len("data: "):]))
        return payloads

    def test_confirmation_is_surfaced_as_an_approval_prompt(self):
        """A confirmation chunk becomes the exact approval question, once."""
        confirmation = {
            "type": "confirmation",
            "role": "computer",
            "content": {"format": "python", "content": "print(1)"},
        }
        self.interpreter.chat = mock.MagicMock(return_value=iter([confirmation]))

        payloads = self._post_chat()

        self.assertEqual(
            [p["choices"][0]["delta"]["content"] for p in payloads],
            ["Do you want to run this code?"],
        )
        self.assertEqual(self.interpreter.chat.call_count, 1)

    def test_confirmation_frame_carries_the_full_chunk_envelope(self):
        """The approval frame is a complete chat.completion.chunk, not just text.

        Clients switch on `object` and read `model`/`id`/`created` to render a
        stream, so the confirmation frame has its own envelope that must match
        the message frames' — the existing test only read its content.
        """
        confirmation = {
            "type": "confirmation",
            "role": "computer",
            "content": {"format": "python", "content": "print(1)"},
        }
        self.interpreter.chat = mock.MagicMock(return_value=iter([confirmation]))

        frame = self._post_chat()[0]

        self.assertEqual(
            set(frame), {"id", "object", "created", "model", "choices"}
        )
        self.assertEqual(frame["object"], "chat.completion.chunk")
        self.assertEqual(frame["model"], "open-interpreter")
        self.assertIsInstance(frame["created"], (int, float))
        self.assertEqual(
            frame["choices"], [{"delta": {"content": "Do you want to run this code?"}}]
        )

    def test_chat_message_frame_carries_the_full_chunk_envelope(self):
        """A streamed chat message frame names itself and its source model.

        This is the second copy of the streaming logic (the post-prompt loop),
        separate from the run-code path, so its envelope has to be pinned too —
        a renamed `choices`/`delta` key would break every non-code client.
        """
        message = {"type": "message", "role": "assistant", "content": "only"}
        self.interpreter.chat = mock.MagicMock(return_value=iter([message]))

        frame = self._post_chat()[0]

        self.assertEqual(
            set(frame), {"id", "object", "created", "model", "choices"}
        )
        self.assertEqual(frame["id"], 0)
        self.assertEqual(frame["object"], "chat.completion.chunk")
        self.assertEqual(frame["model"], "open-interpreter")
        self.assertIsInstance(frame["created"], (int, float))
        self.assertEqual(frame["choices"], [{"delta": {"content": "only"}}])

    def test_chat_code_chunks_stream_as_fenced_frames(self):
        """A code turn streams fence-open, body and fence-close frames in order.

        The chat path reimplements the run-code path's fenced rendering; only the
        run-code copy was covered, so a dropped fence or reordered frame here
        would go unnoticed.
        """
        chunks = [
            {"type": "code", "role": "computer", "format": "python", "start": True},
            {"type": "code", "role": "computer", "format": "python", "content": "x = 1"},
            {"type": "code", "role": "computer", "format": "python", "end": True},
        ]
        self.interpreter.chat = mock.MagicMock(return_value=iter(chunks))

        payloads = self._post_chat()

        self.assertEqual(
            [p["choices"][0]["delta"]["content"] for p in payloads],
            ["```python\n", "x = 1", "\n```\n"],
        )
        self.assertEqual([p["id"] for p in payloads], [0, 1, 2])

    def test_retry_prompt_sequence_is_exact(self):
        """Every silent prompt is tried in the documented order before giving up.

        The prompts escalate as the model stays silent; repeating or dropping one
        would change how many attempts a silent provider gets and what it sees.
        """
        self.interpreter.chat = mock.MagicMock(return_value=iter([]))

        self._post_chat()

        self.assertEqual(
            [call.kwargs["message"] for call in self.interpreter.chat.call_args_list],
            [
                ".",
                "Just say something, anything.",
                "Hello? Answer please.",
                "Are you there?",
                "Can you respond?",
                "Please reply.",
            ],
        )

    def test_silent_prompts_are_retried_until_chunks_arrive(self):
        """Empty prompts yield nothing and fall through to the next prompt."""
        message = {"type": "message", "role": "assistant", "content": "second"}
        self.interpreter.chat = mock.MagicMock(
            side_effect=[iter([]), iter([message])]
        )

        payloads = self._post_chat()

        self.assertEqual(
            [p["choices"][0]["delta"]["content"] for p in payloads], ["second"]
        )
        asked = [
            call.kwargs["message"]
            for call in self.interpreter.chat.call_args_list
        ]
        self.assertEqual(asked[0], ".")
        self.assertEqual(asked[1], "Just say something, anything.")
        for call in self.interpreter.chat.call_args_list:
            self.assertTrue(call.kwargs["stream"])
            self.assertTrue(call.kwargs["display"])

    def test_fully_silent_prompts_yield_an_empty_stream(self):
        """When every prompt stays silent, the stream carries no frames."""
        self.interpreter.chat = mock.MagicMock(return_value=iter([]))

        payloads = self._post_chat()

        self.assertEqual(payloads, [])
        self.assertEqual(self.interpreter.chat.call_count, 6)

    def test_stop_set_during_streaming_suppresses_frames(self):
        """A stop raised while streaming drops the in-flight chunks silently."""
        interpreter = self.interpreter

        def stop_then_yield(**kwargs):
            """Raise the stop flag just before handing over a chunk."""
            interpreter.stop_event.set()
            return iter(
                [{"type": "message", "role": "assistant", "content": "too late"}]
            )

        self.interpreter.chat = mock.MagicMock(side_effect=stop_then_yield)

        payloads = self._post_chat()

        self.assertEqual(payloads, [])


class TestWebSocketAcknowledgedUnsentMessage(TestCase):
    """Pins that an acknowledged unsent message is dropped after sending.

    send_message returns True once the client acks, and send_output pops the
    message on True. A mutation turning the loop's break into a bare return
    makes every acknowledged send report failure, so the message stays queued
    and is re-sent forever — unobservable unless the queue is asserted.
    """

    def test_acknowledged_unsent_message_is_dropped_after_send(self):
        """A pre-acknowledged queued message leaves the queue after delivery."""
        import asyncio

        with mock.patch.dict(os.environ):
            os.environ.pop("INTERPRETER_API_KEY", None)
            from fastapi.testclient import TestClient

            interpreter = AsyncInterpreter()
            interpreter.require_acknowledge = True

            async def hang_output():
                """Park the output poll so only the queued message is sent."""
                await asyncio.sleep(3600)

            interpreter.output = hang_output
            interpreter.input = mock.AsyncMock()
            message = {
                "id": "k1",
                "type": "message",
                "role": "server",
                "content": "hi",
            }
            interpreter.unsent_messages.append(message)
            interpreter.acknowledged_outputs.append("k1")

            client = TestClient(Server(interpreter).app)
            with client.websocket_connect("/") as ws:
                ws.send_text(json.dumps({"auth": "not-checked"}))
                # The queued message may arrive before the auth reply: the
                # send loop starts emitting the moment the socket connects.
                frames = [ws.receive_json(), ws.receive_json()]
                self.assertIn({"auth": True}, frames)
                received = next(f for f in frames if f.get("id") == "k1")
                self.assertEqual(received["content"], "hi")

        self.assertEqual(list(interpreter.unsent_messages), [])


class TestAsyncInterpreterInitialState(TestCase):
    """Pins the exact initial state AsyncInterpreter.__init__ sets up.

    Mutation testing found the constructor's fields survive with any falsy
    stand-in: None, "" and 0 behave the same until later code takes an identity
    or type branch. These pin the documented initial values with identity and
    type checks instead of truthiness.
    """

    def test_control_fields_start_as_none_not_a_falsy_stand_in(self):
        """respond_thread, output_queue and the digest start as None; the queue is a deque.

        Other code compares these with `is None` before use — the input path
        checks `respond_thread is not None` then calls is_alive(), and output()
        creates the queue only when it is None — so "" would take the wrong
        branch and crash, and unsent_messages has to be a real deque so the
        send loop can popleft it.
        """
        from collections import deque

        interpreter = AsyncInterpreter()

        self.assertIsNone(interpreter.respond_thread)
        self.assertIsNone(interpreter.output_queue)
        self.assertIsNone(interpreter.pending_confirmation_digest)
        self.assertIsInstance(interpreter.unsent_messages, deque)

    def test_id_defaults_to_a_timestamp_and_honours_the_env_var(self):
        """The instance id is a numeric timestamp unless INTERPRETER_ID is set.

        The id is handed to clients; dropping the timestamp default would make
        every instance anonymous, and reading a renamed variable would ignore
        the deployment-provided id even when the operator set it.
        """
        with mock.patch.dict(os.environ):
            os.environ.pop("INTERPRETER_ID", None)
            self.assertIsInstance(AsyncInterpreter().id, float)

            os.environ["INTERPRETER_ID"] = "deployment-7"
            self.assertEqual(AsyncInterpreter().id, "deployment-7")

    def test_require_acknowledge_parses_the_env_var_case_insensitively(self):
        """INTERPRETER_REQUIRE_ACKNOWLEDGE gates ack tracking with a case-insensitive match.

        The flag decides whether the server waits for client acknowledgements, so
        a renamed variable or an exact-case comparison would silently disable a
        delivery guarantee the operator asked for.
        """
        with mock.patch.dict(os.environ):
            os.environ.pop("INTERPRETER_REQUIRE_ACKNOWLEDGE", None)
            self.assertIs(AsyncInterpreter().require_acknowledge, False)

            os.environ["INTERPRETER_REQUIRE_ACKNOWLEDGE"] = "TRUE"
            self.assertIs(AsyncInterpreter().require_acknowledge, True)

            os.environ["INTERPRETER_REQUIRE_ACKNOWLEDGE"] = "false"
            self.assertIs(AsyncInterpreter().require_acknowledge, False)


class TestCancelPendingApprovalValue(TestCase):
    """Pins the exact value a cancel records.

    The event set is asserted by the existing tests, but they use assertFalse,
    which a None stand-in also satisfies — only an identity check pins that a
    cancelled approval is specifically denied rather than merely falsy.
    """

    def test_cancel_records_an_exact_false_not_a_falsy_stand_in(self):
        """Cancel leaves _approval_granted as False itself, and sets the event.

        Code distinguishes a cancelled approval from a granted one by the flag's
        value, so a None stand-in would compare false but lose the documented
        False the caller switches on.
        """
        interpreter = AsyncInterpreter()
        interpreter._approval_granted = True

        interpreter._cancel_pending_approval()

        self.assertIs(interpreter._approval_granted, False)
        self.assertTrue(interpreter._approval_event.is_set())


class TestAsyncOutputQueue(TestCase):
    """Pins that output() lazily creates its queue and then reuses it.

    The output() round trip is how responses leave the interpreter, and these
    two lines had no test reaching them: a queue created on every call would
    blackhole anything already queued, and skipping creation would raise instead
    of waiting.
    """

    def test_missing_queue_is_created_and_then_awaited(self):
        """With no queue set, output() installs a janus queue and blocks on it.

        A mutant that assigned None instead of a queue would raise
        AttributeError on the None queue; the timeout below proves it instead
        waits on a real queue.
        """
        import asyncio

        interpreter = AsyncInterpreter()
        interpreter.output_queue = None

        async def exercise():
            """Await output() briefly, then report the queue it installed."""
            try:
                await asyncio.wait_for(interpreter.output(), timeout=0.05)
            except asyncio.TimeoutError:
                pass
            return interpreter.output_queue

        queue = asyncio.run(exercise())

        self.assertIsInstance(queue, janus.Queue)

    def test_existing_queue_is_reused_not_replaced(self):
        """A queued message is returned and the queue object is unchanged.

        Replacing a present queue (the `== None` inversion) would drop the queued
        message and leave the consumer waiting on an empty queue, so both the
        returned value and the object identity are pinned.
        """
        import asyncio

        interpreter = AsyncInterpreter()
        queue = janus.Queue()
        interpreter.output_queue = queue
        queue.sync_q.put("payload")

        result = asyncio.run(asyncio.wait_for(interpreter.output(), timeout=1))

        self.assertEqual(result, "payload")
        self.assertIs(interpreter.output_queue, queue)


class TestServerRunSocketProbe(TestCase):
    """Pins the LAN-IP probe run() performs when bound to 0.0.0.0.

    The probe opens a UDP socket and connects it to a public address to learn the
    machine's outbound IP, then prints it. It only runs for 0.0.0.0, and the
    existing test only checked the printed text, so every socket argument could
    mutate without notice.
    """

    def test_probe_uses_ipv4_udp_and_connects_to_port_80(self):
        """The probe socket is AF_INET/SOCK_DGRAM and connects to the DNS host on :80.

        The socket has to be datagram (a stream connect could block) and the
        address family right for the printed IP to be IPv4; pinning the port
        keeps the documented Google-DNS target the comment describes.
        """
        import contextlib
        import io

        server = Server(AsyncInterpreter())
        server.uvicorn_server.run = mock.Mock()
        server.config.host = "0.0.0.0"
        fake_socket = mock.Mock()
        fake_socket.getsockname.return_value = ("192.168.1.50", 4321)

        with mock.patch(
            "interpreter.core.async_core.socket.socket", return_value=fake_socket
        ) as socket_cls:
            with contextlib.redirect_stdout(io.StringIO()):
                server.run()

        socket_cls.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)
        fake_socket.connect.assert_called_once_with(("8.8.8.8", 80))
        fake_socket.close.assert_called_once_with()


class TestOpenAINonStreamingCompletion(TestCase):
    """Pins the non-streaming branch of POST /openai/chat/completions.

    When a request omits `stream`, the endpoint answers with a single
    chat.completion object rather than an SSE stream. No test exercised that
    branch, so the default stream flag, the default model name and the response
    envelope were all free to mutate — a non-streaming client would get an SSE
    body or a null model.
    """

    def setUp(self):
        """A TestClient with auth open, pointed at a fresh interpreter."""
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("INTERPRETER_API_KEY", None)
        from fastapi.testclient import TestClient

        self.interpreter = AsyncInterpreter()
        self.client = TestClient(Server(self.interpreter).app)

    def _post(self, **extra):
        """POST a plain user turn without `stream` and return the response."""
        body = {"messages": [{"role": "user", "content": "hi"}]}
        body.update(extra)
        return self.client.post("/openai/chat/completions", json=body)

    def _stub_chat(self, content="canned"):
        """Make interpreter.chat return one assistant message with the content."""
        self.interpreter.chat = mock.MagicMock(
            return_value=[{"role": "assistant", "type": "message", "content": content}]
        )

    def test_absent_stream_returns_a_single_completion_object(self):
        """A request without `stream` gets one JSON chat.completion object.

        The default must stay False: a mutated default of True would return an
        SSE body where the client expects JSON, and the model name has to be the
        documented default rather than null.
        """
        self._stub_chat("canned")

        response = self._post()

        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers["content-type"])
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["id"], "200")
        self.assertEqual(payload["model"], "default-model")
        self.assertIsInstance(payload["created"], (int, float))
        self.assertEqual(
            payload["choices"],
            [{"message": {"role": "assistant", "content": "canned"}}],
        )

    def test_request_model_is_echoed_back(self):
        """An explicit model is passed through to the response untouched."""
        self._stub_chat("x")

        response = self._post(model="gpt-4o-mini")

        self.assertEqual(response.json()["model"], "gpt-4o-mini")


class TestAsyncInputCommandGaps(TestCase):
    """Pins command-handling details the existing command tests leave open.

    The existing go/stop tests cover the happy paths, so pinned here is the exact
    bookkeeping: which messages are consumed, how a digest is split, that a bare
    "go" approves any pending payload, that an end chunk really starts a respond
    thread, and the exact error chunk a refused command emits.
    """

    def setUp(self):
        """An interpreter with a mocked output queue and a live mock respond thread."""
        self.interpreter = AsyncInterpreter()
        self.interpreter.auto_run = False
        self.interpreter.output_queue = mock.MagicMock()
        self.interpreter.output_queue.sync_q = mock.MagicMock()
        self.interpreter.respond_thread = mock.MagicMock()
        self.interpreter.respond_thread.is_alive.return_value = True

    def _feed_command(self, command):
        """Feed start/content/end chunks for one user command."""
        import asyncio

        async def run():
            """Drive interpreter.input over the three command chunks."""
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

    def _puts(self):
        """The raw chunk objects put on the sync output queue, in order."""
        return [
            call.args[0]
            for call in self.interpreter.output_queue.sync_q.put.call_args_list
        ]

    def _errors(self):
        """The error-type chunks put on the sync output queue, in order."""
        return [chunk for chunk in self._puts() if chunk.get("type") == "error"]

    def test_bare_go_approves_any_pending_confirmation(self):
        """A bare "go" grants the pending approval without a digest check.

        The digest is optional for clients that accept whatever is pending, so an
        empty-string stand-in for the "no digest" sentinel must not turn "go" into
        a refusal.
        """
        payload = {"format": "python", "content": "print(1)"}
        self.interpreter.pending_confirmation = payload
        self.interpreter.pending_confirmation_digest = confirmation_digest(payload)

        self._feed_command("go")

        self.assertEqual(self._errors(), [])
        self.assertIs(self.interpreter._approval_granted, True)

    def test_go_digest_keeps_everything_after_the_first_colon(self):
        """go:<digest> takes the whole remainder, not the first or last field.

        Digests are hex today, but the split must not truncate or reorder the
        token: a client echoing a digest containing a colon would otherwise be
        refused or approved against the wrong part.
        """
        self.interpreter.pending_confirmation = {"format": "python", "content": "x"}
        self.interpreter.pending_confirmation_digest = "aa:bb:cc"

        self._feed_command("go:aa:bb:cc")

        self.assertEqual(self._errors(), [])
        self.assertIs(self.interpreter._approval_granted, True)

    def test_handling_a_command_removes_only_that_command(self):
        """The command chunk is popped, leaving earlier transcript entries intact.

        Dropping extra messages would silently delete the user's previous turn
        every time a go/stop command is processed.
        """
        kept = {"role": "user", "type": "message", "content": "earlier turn"}
        self.interpreter.messages = [kept]
        self.interpreter.pending_confirmation = {"format": "python", "content": "x"}
        self.interpreter.pending_confirmation_digest = "d"

        self._feed_command("go:d")

        self.assertEqual(self.interpreter.messages, [kept])

    def test_refused_command_emits_the_exact_server_error_chunk(self):
        """A mismatched approval emits one server/error chunk plus the complete marker.

        Clients key off `role`, `type` and the complete marker to close the turn,
        so a renamed field would leave them waiting; the chunk shape is the
        contract, not just the message text.
        """
        self._feed_command("go:whatever")

        self.assertEqual(
            self._puts(),
            [
                {
                    "role": "server",
                    "type": "error",
                    "content": "No pending code approval matches that request.",
                },
                complete_message,
            ],
        )

    def test_end_chunk_starts_a_respond_thread_with_run_code_none(self):
        """An end chunk spawns Thread(target=respond, args=(None,)) and starts it.

        run_code begins as None so respond() defaults it to auto_run; dropping
        the target or the args would spawn a thread that never responds, or one
        bound to the wrong flag.
        """
        import asyncio

        self.interpreter.messages = [
            {"role": "user", "type": "message", "content": "turn"}
        ]
        self.interpreter.respond_thread = None

        with mock.patch(
            "interpreter.core.async_core.threading.Thread"
        ) as thread_cls:
            asyncio.run(
                self.interpreter.input({"role": "user", "type": "message", "end": True})
            )

        kwargs = thread_cls.call_args.kwargs
        # Bound methods are recreated per attribute access, so compare by equality
        # (same function and instance) rather than identity.
        self.assertEqual(kwargs["target"], self.interpreter.respond)
        self.assertEqual(kwargs["args"], (None,))
        thread_cls.return_value.start.assert_called_once_with()


class TestAsyncRespondGrantedTurnSecondConfirmation(TestCase):
    """Pins that only the first confirmation under a granted turn runs silently.

    respond(run_code=True) is the "yes" turn: the first confirmation is the user
    authorising execution, but a second confirmation in the same turn is new code
    and must still pause. A mutation that kept the grant alive would run every
    later confirmation without approval.
    """

    def test_second_confirmation_under_a_granted_turn_still_pauses(self):
        """The first confirmation is consumed, the second parks for approval.

        run_code must drop back to False after the first confirmation; leaving it
        truthy would auto-approve the second payload, executing code the user
        never saw.
        """
        first = {
            "type": "confirmation",
            "role": "computer",
            "content": {"format": "python", "content": "first"},
        }
        second = {
            "type": "confirmation",
            "role": "computer",
            "content": {"format": "python", "content": "second"},
        }
        interpreter = AsyncInterpreter()
        interpreter.auto_run = False
        interpreter.output_queue = mock.MagicMock(sync_q=mock.MagicMock())

        def store():
            """Yield the two confirmations in one response stream."""
            yield first
            yield second

        with mock.patch.object(interpreter, "_respond_and_store", store):
            worker = threading.Thread(
                target=interpreter.respond, args=(True,), daemon=True
            )
            worker.start()
            # Wait for the second payload specifically, and only once respond()
            # has finished parking on it. It sets pending_confirmation before it
            # clears _approval_event and then blocks, so approving as soon as the
            # field is non-None can land before that clear and have the grant
            # erased -- which would wedge the worker. complete_message is the
            # signal that the clear has already happened.
            wait_until(
                lambda: interpreter.pending_confirmation
                == {"format": "python", "content": "second"}
                and complete_message
                in [call.args[0] for call in interpreter.output_queue.sync_q.put.call_args_list],
                "respond() never parked on the second confirmation",
                fail=self.fail,
            )

            self.assertEqual(
                interpreter.pending_confirmation, {"format": "python", "content": "second"}
            )
            self.assertTrue(interpreter._approve_pending_confirmation())
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive())


class TestConfirmationDigestValues(TestCase):
    """Pins the exact digest a confirmation payload hashes to.

    The digest is the approval token: the UI shows it, the client echoes it back
    as `go:<digest>`, and a mismatch refuses execution. Existing tests only
    checked that a digest is stable and that a wrong string is rejected; nothing
    pinned the digest's inputs, so dropping the language or the code from the
    hash left approvals bound to half the payload.
    """

    def test_digest_hashes_language_and_code_together(self):
        """The digest is sha256 of "language\\0code" over the payload's fields.

        Pinning the exact value fixes the on-the-wire approval token clients
        round-trip; an omitted field or changed separator would make different
        code payloads share a digest (or valid ones stop matching).
        """
        payload = {"format": "python", "content": "print(1)"}

        self.assertEqual(
            confirmation_digest(payload),
            hashlib.sha256(b"python\x00print(1)").hexdigest(),
        )

    def test_language_and_code_each_change_the_digest(self):
        """Changing either field changes the digest, and missing fields hash as "".

        A payload hashed without its language or without its code would collide
        with a different payload's digest, letting one approval authorise code
        the reviewer never saw.
        """
        base = {"format": "python", "content": "print(1)"}

        self.assertNotEqual(
            confirmation_digest(base),
            confirmation_digest({"format": "java", "content": "print(1)"}),
        )
        self.assertNotEqual(
            confirmation_digest(base),
            confirmation_digest({"format": "python", "content": "print(2)"}),
        )
        self.assertEqual(
            confirmation_digest({}),
            confirmation_digest({"format": "", "content": ""}),
        )

