"""Tests for JupyterLanguage.run() and the iopub listener's error handling.

Two failure paths that decide what the user sees when things go wrong: the
listener's bounded retry, and `execute`'s conversion of an exception into an
output chunk. Neither is about the happy path, and neither was covered.
"""

import queue
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.computer.terminal.languages.jupyter_language import JupyterLanguage


def _language(**attrs):
    """A JupyterLanguage with a stub kernel, built without starting a real one."""
    lang = JupyterLanguage.__new__(JupyterLanguage)
    lang.computer = SimpleNamespace(
        interpreter=SimpleNamespace(
            stop_event=SimpleNamespace(is_set=lambda: False),
            save_skills=False,
            messages=[],
            llm=SimpleNamespace(model="m", api_key=None),
        )
    )
    lang.km = SimpleNamespace(interrupt_kernel=mock.Mock())
    lang.kc = SimpleNamespace(
        execute=mock.Mock(),
        input=mock.Mock(),
        is_alive=lambda: True,   # run() spins until the kernel answers
        iopub_channel=SimpleNamespace(),
    )
    lang.finish_flag = False
    lang.last_output_time = time.time()
    lang.last_output_message_time = time.time()
    for key, value in attrs.items():
        setattr(lang, key, value)
    return lang


class _QueueAfterEmpty(queue.Queue):
    """A queue whose first get raises Empty, as if an item landed just after."""

    def __init__(self, item):
        super().__init__()
        self.put(item)
        self.gets = 0

    def get(self, block=True, timeout=None):
        self.gets += 1
        if self.gets == 1:
            raise queue.Empty
        return super().get(block=block, timeout=timeout)


def test_a_persistent_listener_failure_is_re_raised_after_the_retry_budget(monkeypatch):
    """A kernel that keeps erroring surfaces eventually instead of spinning forever.

    The listener catches every exception, decrements `max_retries` and continues.
    If that never terminated, a broken kernel would hang the listener thread and
    the user would wait on output that could never arrive — a silent hang rather
    than an error.

    The listener runs on its own thread, so `threading.Thread` is replaced with a
    stub whose `start()` runs the target inline. That makes the retry loop
    observable and keeps a real runaway thread out of the suite.
    """
    lang = _language()
    attempts = []
    raised = {}

    def always_fails(timeout=None):
        attempts.append(1)
        raise RuntimeError("kernel is gone")

    lang.kc.iopub_channel.get_msg = always_fails

    class InlineThread:
        def __init__(self, target):
            self._target = target
            self.daemon = False

        def start(self):
            try:
                self._target()
            except BaseException as error:  # noqa: BLE001 - recorded, then re-asserted
                raised["error"] = error

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(
        "interpreter.core.computer.terminal.languages.jupyter_language.threading.Thread",
        InlineThread,
    )

    lang._execute_code("print(1)", queue.Queue())

    assert len(attempts) == 101, (
        f"expected 100 retries plus the initial attempt, got {len(attempts)}"
    )
    assert isinstance(raised.get("error"), RuntimeError), (
        f"the listener must re-raise once the budget is spent, got {raised.get('error')!r}"
    )
    assert "kernel is gone" in str(raised["error"])


def test_a_preprocessing_failure_falls_back_to_the_raw_code(monkeypatch):
    """Code that cannot be preprocessed still runs, unmodified.

    Preprocessing exists only to inject active-line markers, so a failure there
    is ours and must not stop the user's code from running. The comment in the
    source says as much; this pins it.
    """
    lang = _language()

    def broken_preprocess(_code):
        raise ValueError("preprocessing exploded")

    executed = {}

    def capture_execute(code, message_queue):
        executed["code"] = code

    monkeypatch.setattr(lang, "preprocess_code", broken_preprocess)
    monkeypatch.setattr(lang, "_execute_code", capture_execute)
    monkeypatch.setattr(lang, "_capture_output", lambda q: iter([]))

    assert list(lang.run("x = 1")) == []
    assert executed["code"] == "x = 1", "the original code should run unchanged"


def test_preprocessed_code_is_used_when_preprocessing_succeeds(monkeypatch):
    """The complement: a successful preprocess is what actually gets executed."""
    lang = _language()
    executed = {}

    monkeypatch.setattr(lang, "preprocess_code", lambda _code: "PROCESSED")
    monkeypatch.setattr(lang, "_execute_code", lambda code, q: executed.update(code=code))
    monkeypatch.setattr(lang, "_capture_output", lambda q: iter([]))

    list(lang.run("x = 1"))

    assert executed["code"] == "PROCESSED"


def test_an_exception_during_execution_is_reported_as_output(monkeypatch):
    """A failure inside execution becomes an output chunk, not a lost turn.

    The user sees the traceback in the output stream. Without this the generator
    would raise into the caller mid-stream and the turn would end with no
    explanation of what happened.
    """
    lang = _language()
    monkeypatch.setattr(lang, "preprocess_code", lambda _code: "PROCESSED")

    def explode(code, message_queue):
        raise ValueError("execution exploded")

    monkeypatch.setattr(lang, "_execute_code", explode)

    chunks = list(lang.run("x = 1"))

    assert len(chunks) == 1, f"expected exactly one output chunk, got {chunks}"
    assert chunks[0]["type"] == "console"
    assert chunks[0]["format"] == "output"
    assert "Traceback" in chunks[0]["content"]
    assert "ValueError: execution exploded" in chunks[0]["content"]


def test_generator_exit_is_passed_on_rather_than_swallowed(monkeypatch):
    """Closing the output generator mid-stream propagates instead of yielding a traceback.

    `execute` re-raises GeneratorExit explicitly. A bare `except:` would instead
    try to yield, which Python turns into "generator ignored GeneratorExit" — the
    consumer would get an error chunk for something it did by closing the stream.
    """
    lang = _language()
    monkeypatch.setattr(lang, "preprocess_code", lambda _code: "PROCESSED")
    monkeypatch.setattr(lang, "_execute_code", lambda code, q: None)
    monkeypatch.setattr(
        lang, "_capture_output", lambda q: (_ for _ in ()).throw(GeneratorExit())
    )

    generator = lang.run("x = 1")
    with pytest.raises(GeneratorExit):
        next(generator)


def test_an_abandoned_listener_is_joined_and_cannot_steal_the_next_run(monkeypatch):
    """A still-alive listener from an abandoned run is stopped before the next request.

    If a caller closes a run() generator early, its listener thread may still be
    blocked on the shared iopub channel. Starting a new request without stopping
    it lets the old listener consume the new run's messages (get_msg removes
    them from the channel) and set finish_flag on the new run's idle status.
    Each listener is therefore bound to its own request id, and _execute_code
    stops and joins a lingering listener before sending the next request.
    """
    lang = _language()
    lang.kc.execute = mock.Mock(side_effect=["id-1", "id-2"])

    parked = {}

    class ParkedThread:
        """A first-run listener that never exits, like an abandoned generator's."""

        def __init__(self, target):
            parked["target"] = target
            parked["joined"] = False

        def start(self):
            pass  # stays blocked on the channel, as an abandoned listener would

        def is_alive(self):
            return True

        def join(self, timeout=None):
            parked["joined"] = True

    monkeypatch.setattr(
        "interpreter.core.computer.terminal.languages.jupyter_language.threading.Thread",
        ParkedThread,
    )
    lang._execute_code("print(1)", queue.Queue())

    def message(parent_id, kind, **content):
        return {
            "parent_header": {"msg_id": parent_id},
            "header": {"msg_type": kind},
            "msg_type": kind,
            "content": content,
        }

    script = [
        message("id-1", "stream", text="stale output\n"),
        message("id-2", "stream", text="new output\n"),
        message("id-2", "status", execution_state="idle"),
    ]
    lang.kc.iopub_channel.get_msg = lambda timeout=None: script.pop(0)

    raised = {}

    class InlineThread:
        def __init__(self, target):
            self._target = target

        def start(self):
            try:
                self._target()
            except BaseException as error:  # noqa: BLE001 - recorded, then re-asserted
                raised["error"] = error

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(
        "interpreter.core.computer.terminal.languages.jupyter_language.threading.Thread",
        InlineThread,
    )
    second_queue = queue.Queue()
    lang._execute_code("print(2)", second_queue)

    assert parked["joined"], "the abandoned listener must be joined first"
    assert "error" not in raised
    assert lang._execution_id == "id-2"
    chunks = []
    while not second_queue.empty():
        chunks.append(second_queue.get())
    assert chunks == [
        {"type": "console", "format": "output", "content": "new output\n"}
    ], f"the stale run's message must be skipped, got {chunks}"


def test_the_extra_joins_stop_as_soon_as_the_stuck_listener_exits(monkeypatch):
    """Once the stuck listener finally exits, the bounded join loop stops early.

    The extra joins exist to give a listener stuck inside a slow model call time
    to notice the stop flag. As soon as it is gone the loop must break rather
    than burn the remaining waits.
    """
    lang = _language()
    lang.kc.execute = mock.Mock(return_value="id-1")

    class EventuallyDead:
        """Alive for the pre-check and the first post-join check, then dead."""

        def __init__(self):
            self.alive_calls = 0
            self.joins = 0

        def is_alive(self):
            self.alive_calls += 1
            return self.alive_calls <= 2

        def join(self, timeout=None):
            self.joins += 1

    stuck = EventuallyDead()
    lang.listener_thread = stuck

    class NoopThread:
        def __init__(self, target):
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(
        "interpreter.core.computer.terminal.languages.jupyter_language.threading.Thread",
        NoopThread,
    )

    lang._execute_code("print(1)", queue.Queue())

    # One join before the loop, one inside it that observed the exit and broke.
    assert stuck.joins == 2


def test_a_listener_from_a_superseded_generation_stops_consuming(monkeypatch):
    """A listener whose generation is stale returns without reading the channel.

    The join cannot always stop a stuck listener, so each listener also checks
    its generation: once a newer request starts, the old one exits instead of
    stealing the new run's messages.
    """
    lang = _language()
    lang.kc.execute = mock.Mock(side_effect=["id-1", "id-2"])

    targets = []

    class DeferredThread:
        """Captures the listener target without running it."""

        def __init__(self, target):
            targets.append(target)

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(
        "interpreter.core.computer.terminal.languages.jupyter_language.threading.Thread",
        DeferredThread,
    )

    lang._execute_code("print(1)", queue.Queue())  # generation 1
    lang._execute_code("print(2)", queue.Queue())  # generation 2 supersedes it

    lang.kc.iopub_channel.get_msg = mock.Mock(
        side_effect=AssertionError("a stale listener must not read the channel")
    )

    # Running the superseded listener returns at the generation check.
    targets[0]()
    lang.kc.iopub_channel.get_msg.assert_not_called()


def _collect(generator, timeout=15):
    """Drain a generator in a thread so a hang fails the test, not the suite."""
    result = {}

    def worker():
        try:
            result["chunks"] = list(generator)
        except Exception as exc:  # pragma: no cover - surfaced by the assertion
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "generator hung instead of returning"
    assert "error" not in result, result.get("error")
    return result["chunks"]


def test_kernel_is_alive_uses_the_process_not_the_client():
    """Liveness must come from the manager: the client stays up after a hard exit."""
    lang = _language()
    lang.km = SimpleNamespace(is_alive=lambda: False)
    assert lang._kernel_is_alive() is False

    lang.km = SimpleNamespace(is_alive=lambda: True)
    assert lang._kernel_is_alive() is True


def test_a_dead_kernel_is_restarted_before_the_next_run(monkeypatch):
    """os._exit(0) must not leave the next run executing against a dead kernel."""
    lang = _language()
    lang.kc.stop_channels = mock.Mock()
    # The manager can be wedged too; a failed shutdown must not block the restart.
    dead_km = SimpleNamespace(
        is_alive=lambda: False,
        shutdown_kernel=mock.Mock(side_effect=RuntimeError("already gone")),
    )
    lang.km = dead_km
    restarted = {}

    def fake_init(_self, computer):
        restarted["computer"] = computer
        lang.km = SimpleNamespace(is_alive=lambda: True, shutdown_kernel=mock.Mock())

    monkeypatch.setattr(JupyterLanguage, "__init__", fake_init)
    lang._execute_code = lambda code, message_queue: None
    lang._capture_output = lambda message_queue: iter(
        [{"type": "console", "format": "output", "content": "ok"}]
    )

    chunks = list(lang.run("x = 1"))

    assert restarted["computer"] is lang.computer
    lang.kc.stop_channels.assert_called_once_with()
    dead_km.shutdown_kernel.assert_called_once_with(now=True)
    assert chunks == [{"type": "console", "format": "output", "content": "ok"}]


def test_a_listener_reports_a_dead_kernel_and_stops():
    """A hard exit is reported instead of waiting for an idle that cannot arrive.

    The channels outlive the process, so the iopub reader would otherwise spin
    until the user interrupts, and the consumer has no exit condition.
    """
    lang = _language()
    lang.km = SimpleNamespace(is_alive=lambda: False, interrupt_kernel=mock.Mock())
    lang.kc.execute = mock.Mock(return_value="msg-1")
    message_queue = queue.Queue()

    lang._execute_code("pass", message_queue)

    lang.listener_thread.join(timeout=5)
    assert not lang.listener_thread.is_alive()
    assert lang.finish_flag is True
    assert "kernel exited" in message_queue.get_nowait()["content"]


def test_capture_output_stops_when_the_listener_dies_without_finishing():
    """A listener that dies without setting finish_flag must not hang the consumer."""
    lang = _language()
    listener = threading.Thread(target=lambda: None)
    listener.start()
    listener.join()
    lang.listener_thread = listener
    lang.finish_flag = False

    chunks = _collect(lang._capture_output(queue.Queue()))

    assert chunks == []
    assert lang.finish_flag is True


def test_kernel_is_alive_fails_open_when_the_manager_is_unavailable():
    """A manager that raises, or none at all, must not block a run from starting."""
    lang = _language()

    def boom():
        raise RuntimeError("kernel manager is gone")

    lang.km = SimpleNamespace(is_alive=boom)
    assert lang._kernel_is_alive() is True

    bare = JupyterLanguage.__new__(JupyterLanguage)
    assert bare._kernel_is_alive() is True


def test_kernel_is_alive_reports_a_failing_probe_only_when_verbose(capsys):
    """A consistently failing liveness probe must be diagnosable under verbose.

    The probe fails open so a transient error cannot throw away a live session,
    but a silent fail-open makes the original hang indistinguishable from a
    healthy kernel. Verbose output is the only place that difference is visible.
    """
    lang = _language()
    lang.computer.verbose = False

    def boom():
        raise RuntimeError("kernel manager is gone")

    lang.km = SimpleNamespace(is_alive=boom)

    assert lang._kernel_is_alive() is True
    assert capsys.readouterr().out == ""

    lang.computer.verbose = True
    assert lang._kernel_is_alive() is True
    assert "kernel manager is gone" in capsys.readouterr().out


def test_a_kernel_that_cannot_be_restarted_reports_it(monkeypatch):
    """When the restart fails, the run says so instead of executing anyway."""
    lang = _language()
    lang.km = SimpleNamespace(is_alive=lambda: False, shutdown_kernel=mock.Mock())
    monkeypatch.setattr(JupyterLanguage, "__init__", lambda _self, computer: None)

    chunks = list(lang.run("x = 1"))

    assert chunks == [
        {
            "type": "console",
            "format": "output",
            "content": "The Python kernel could not be restarted.",
        }
    ]


def test_run_waits_only_briefly_for_channels_that_never_come_up(monkeypatch):
    """A client whose channels never open ends the run instead of spinning.

    The old wait was `while not kc.is_alive(): sleep`, with no deadline, so a
    kernel that failed to start hung the run before it ever yielded.
    """
    monkeypatch.setattr(
        "interpreter.core.computer.terminal.languages.jupyter_language.KERNEL_WAIT_SECONDS",
        0.05,
    )
    lang = _language()
    lang.kc.is_alive = lambda: False
    lang._execute_code = lambda code, message_queue: None
    lang._capture_output = lambda message_queue: iter(
        [{"type": "console", "format": "output", "content": "ok"}]
    )

    chunks = _collect(lang.run("x = 1"))

    assert chunks == [{"type": "console", "format": "output", "content": "ok"}]


def test_capture_output_stops_when_the_kernel_is_dead():
    """A consumer waiting on a dead kernel stops and still reports the death.

    The listener may not have enqueued its notice before the consumer decides
    the queue is empty, so the consumer delivers the notice itself rather than
    ending the run silently.
    """
    lang = _language()
    lang.km = SimpleNamespace(is_alive=lambda: False)
    lang.listener_thread = threading.Thread(target=lambda: None)
    lang.finish_flag = False

    chunks = _collect(lang._capture_output(queue.Queue()))

    assert chunks == [
        {
            "type": "console",
            "format": "output",
            "content": "The Python kernel exited. It will restart on the next run.",
        }
    ]
    assert lang.finish_flag is True


def test_capture_output_drains_output_that_arrives_as_the_kernel_dies():
    """Output queued before the death check is still shown to the user.

    The listener can put a chunk between the consumer's timed-out get and the
    liveness check, so the dead-kernel branch drains what is there rather than
    dropping the run's last output.
    """

    lang = _language()
    lang.km = SimpleNamespace(is_alive=lambda: False)
    lang.listener_thread = threading.Thread(target=lambda: None)
    lang.finish_flag = False
    pending = {"type": "console", "format": "output", "content": "last words"}

    chunks = _collect(lang._capture_output(_QueueAfterEmpty(pending)))

    # The chunk is preserved, and the death is still reported: the listener
    # never got the chance to enqueue its own notice here.
    assert chunks == [
        pending,
        {
            "type": "console",
            "format": "output",
            "content": "The Python kernel exited. It will restart on the next run.",
        },
    ]
    assert lang.finish_flag is True


def test_a_restart_that_raises_is_reported_and_retried(monkeypatch):
    """An `__init__` failure must surface as output, not escape run().

    A failed restart used to propagate out of `run()` before it could say what
    went wrong, and a manager left half-built could be trusted by the next run.
    The failure is reported, the incomplete kernel is not reused, and the next
    run tries again.
    """
    lang = _language()
    lang.km = SimpleNamespace(is_alive=lambda: False, shutdown_kernel=mock.Mock())
    lang.kc.stop_channels = mock.Mock()
    lang._kernel_ready = False

    def exploding_init(_self, computer):
        raise RuntimeError("no kernelspec")

    monkeypatch.setattr(JupyterLanguage, "__init__", exploding_init)

    chunks = list(lang.run("x = 1"))

    assert chunks == [
        {
            "type": "console",
            "format": "output",
            "content": "The Python kernel could not be restarted: no kernelspec",
        }
    ]
    assert lang._kernel_ready is False

    attempts = {"n": 0}

    def recovering_init(_self, computer):
        attempts["n"] += 1
        lang.km = SimpleNamespace(is_alive=lambda: True, shutdown_kernel=mock.Mock())
        lang.kc = SimpleNamespace(
            execute=mock.Mock(),
            input=mock.Mock(),
            is_alive=lambda: True,
            iopub_channel=SimpleNamespace(),
        )
        lang._kernel_ready = True

    monkeypatch.setattr(JupyterLanguage, "__init__", recovering_init)
    lang._execute_code = lambda code, message_queue: None
    lang._capture_output = lambda message_queue: iter(
        [{"type": "console", "format": "output", "content": "ok"}]
    )

    assert list(lang.run("x = 1")) == [
        {"type": "console", "format": "output", "content": "ok"}
    ]
    assert attempts["n"] == 1


def test_init_gives_up_when_the_new_kernel_never_starts(monkeypatch):
    """A kernel whose channels never open raises instead of waiting forever."""
    monkeypatch.setattr(
        "interpreter.core.computer.terminal.languages.jupyter_language.KERNEL_WAIT_SECONDS",
        0.05,
    )
    manager = mock.Mock()
    manager.is_alive.return_value = True  # the process keeps running
    manager.client.return_value.is_alive.return_value = False  # channels never open

    with mock.patch(
        "interpreter.core.computer.terminal.languages.jupyter_language.KernelManager",
        return_value=manager,
    ):
        with pytest.raises(RuntimeError, match="channels never came up"):
            JupyterLanguage(SimpleNamespace())


def test_init_reports_a_kernel_that_exits_before_its_channels_open(monkeypatch):
    """A replacement process that dies immediately is reported, not waited on.

    The wait would otherwise run to its deadline for a process that is already
    gone, and a later run would inherit the unusable manager.
    """
    monkeypatch.setattr(
        "interpreter.core.computer.terminal.languages.jupyter_language.KERNEL_WAIT_SECONDS",
        30,
    )
    manager = mock.Mock()
    manager.is_alive.return_value = False
    manager.client.return_value.is_alive.return_value = False

    with mock.patch(
        "interpreter.core.computer.terminal.languages.jupyter_language.KernelManager",
        return_value=manager,
    ):
        with pytest.raises(RuntimeError, match="channels never came up"):
            JupyterLanguage(SimpleNamespace())


def test_capture_output_does_not_repeat_the_exit_notice():
    """A notice read through the normal queue path is not emitted a second time.

    The listener queues the notice; the consumer yields it and then takes the
    dead-kernel branch on its next timeout. That branch must not add another
    copy, or non-streaming output repeats the same sentence.
    """
    lang = _language()
    lang.km = SimpleNamespace(is_alive=lambda: False)
    lang.listener_thread = threading.Thread(target=lambda: None)
    notice = {
        "type": "console",
        "format": "output",
        "content": "The Python kernel exited. It will restart on the next run.",
    }
    queued = queue.Queue()
    queued.put(notice)

    chunks = _collect(lang._capture_output(queued))

    assert chunks == [notice]
    assert lang.finish_flag is True


def test_capture_output_and_drain_do_not_repeat_the_exit_notice():
    """A notice drained after the dead-kernel check is yielded once, not twice.

    The listener can enqueue its notice just after the consumer's timeout, so
    the drain loop has to recognize it and suppress the fallback copy the same
    way the normal queue path does.
    """
    lang = _language()
    lang.km = SimpleNamespace(is_alive=lambda: False)
    lang.listener_thread = threading.Thread(target=lambda: None)
    notice = {
        "type": "console",
        "format": "output",
        "content": "The Python kernel exited. It will restart on the next run.",
    }
    queued = _QueueAfterEmpty(notice)

    chunks = _collect(lang._capture_output(queued))

    assert chunks == [notice]
    assert lang.finish_flag is True


def test_a_restart_without_a_client_shuts_down_only_the_manager(monkeypatch):
    """A half-built kernel may have no client, so the restart must not assume one.

    ``__init__`` can fail after the manager exists but before a client is
    created; the restart path then has to shut the manager down without
    touching a missing ``kc``, rather than raising AttributeError.
    """
    lang = _language()
    lang.kc = None
    lang._kernel_ready = False
    dead_km = SimpleNamespace(is_alive=lambda: False, shutdown_kernel=mock.Mock())
    lang.km = dead_km
    restarted = {}

    def fake_init(_self, computer):
        restarted["computer"] = computer
        _self._kernel_ready = True

    monkeypatch.setattr(JupyterLanguage, "__init__", fake_init)

    lang._ensure_kernel()

    assert restarted["computer"] is lang.computer
    dead_km.shutdown_kernel.assert_called_once_with(now=True)
    assert lang._kernel_restart_error is None
    assert lang._kernel_ready is True
