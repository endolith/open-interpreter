"""Tests for JupyterLanguage.run() and the iopub listener's error handling.

Two failure paths that decide what the user sees when things go wrong: the
listener's bounded retry, and `execute`'s conversion of an exception into an
output chunk. Neither is about the happy path, and neither was covered.
"""

import queue
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
