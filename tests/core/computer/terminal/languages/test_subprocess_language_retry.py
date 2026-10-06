"""Tests for SubprocessLanguage's write-retry-restart loop.

Every non-Python code block runs through this: bash, JavaScript, Java, R, HTML.
When writing to the child process's stdin fails — a broken pipe because the
process exited, a full pipe buffer — the loop restarts the process and retries,
up to a bounded number of times, then gives up with a message rather than hanging
or failing silently.

That recovery is invisible in normal operation and load-bearing when it triggers,
which is why it had no test.
"""

import queue
import threading
import time
from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.terminal.languages.subprocess_language import (
    SubprocessLanguage,
)


def _language(start_cmd=("bash",)):
    """A SubprocessLanguage with a fake Popen and no real subprocess."""
    lang = SubprocessLanguage.__new__(SubprocessLanguage)
    lang.start_cmd = list(start_cmd)
    lang.verbose = False
    # Real queue and Event, not mocks. `run`'s collection loop only ends when
    # `output_queue.get(timeout=...)` raises Empty *and* `done` is set; a Mock
    # never raises, so the loop spins forever.
    lang.output_queue = queue.Queue()
    lang.done = threading.Event()
    lang.process = None
    lang.start_process = mock.Mock()
    # The fake child is already finished, so the loop drains and exits.
    lang.done.set()
    return lang


def _with_process(lang, *, write_errors=0):
    """Give the language a child process whose stdin fails `write_errors` times.

    A failing write models a broken pipe: the child is gone, or its stdin cannot
    accept more. `BrokenPipeError` is what CPython raises for that.

    `start_process` is stubbed to install a *working* stdin, which is what the
    real one does. Without that the same failing stdin is reused for every retry
    and the run always exhausts its budget — the restart becomes untestable, and
    it would look like the product never recovers.
    """
    def attach(start_side_effect=None):
        stdin = mock.Mock()
        if start_side_effect:
            stdin.write.side_effect = start_side_effect
        lang.process = SimpleNamespace(stdin=stdin, stdout=None, stderr=None)

    attach([BrokenPipeError("pipe closed")] * write_errors)

    def restart():
        # A restarted child gets a fresh, working stdin; further failures are
        # requested explicitly by the test through `restart_failures`.
        attach(restart_failures.pop(0) if restart_failures else None)
        _finish_when_written(lang)

    def _finish_when_written(lang):
        """Make the run terminate once code is accepted.

        `run` clears `done` before writing, then collects output until
        `output_queue.get(timeout=...)` is empty *and* `done` is set. Nothing sets
        `done` here because the real `handle_stream_output` thread is not running,
        so the loop would block forever. Queueing a marker and setting `done` from
        a watcher thread gives the collection loop something to drain and an exit
        condition — which is what the fake child "producing output then exiting"
        would do.
        """
        def watch():
            for _ in range(200):
                if lang.process is not None and lang.process.stdin.write.call_count:
                    break
                time.sleep(0.005)
            lang.output_queue.put({"type": "console", "format": "output", "content": "done"})
            lang.done.set()

        threading.Thread(target=watch, daemon=True).start()

    restart_failures = []
    lang.start_process = mock.Mock(side_effect=restart)
    lang.restart_failures = restart_failures
    _finish_when_written(lang)
    return lang


def test_a_failing_write_restarts_the_process_and_retries():
    """One failed write is recovered from invisibly — no message, code still runs.

    The first failure is deliberately silent: it happens often in practice (the
    comment in the source blames applescript), and surfacing it would mean every
    bash block reported a spurious error.
    """
    lang = _with_process(_language(), write_errors=1)

    chunks = list(lang.run("echo hi"))

    assert lang.start_process.call_count == 1, "the process should be restarted once"
    assert all("Retrying" not in str(c.get("content", "")) for c in chunks), (
        f"the first failure should be silent, got {chunks}"
    )


def test_the_written_code_is_retried_against_the_new_process():
    """After a restart the code is written again, so the execution still happens.

    Restarting without re-writing would clear the failure but silently drop the
    user's code — the worst outcome, because the transcript would show a block
    that never ran.
    """
    lang = _with_process(_language(), write_errors=1)
    failing_stdin = lang.process.stdin

    list(lang.run("echo hi"))

    assert lang.start_process.call_count == 1
    assert failing_stdin.write.call_count == 1, "the first attempt tries once"
    # The restarted process is the one that receives the code.
    assert lang.process is not failing_stdin, "a new process should be attached"
    # run() appends a newline before writing, so the payload is "echo hi\n".
    assert lang.process.stdin.write.call_args[0][0] == "echo hi\n", (
        "the code must be re-written to the restarted process"
    )



def test_repeated_failures_report_a_retry_message():
    """A second failure is surfaced, because it is no longer routine."""
    lang = _with_process(_language(), write_errors=1)
    lang.restart_failures.append([BrokenPipeError("pipe closed again")])

    chunks = list(lang.run("echo hi"))

    retry_messages = [c for c in chunks if "Retrying" in str(c.get("content", ""))]
    assert retry_messages, f"expected a retry message, got {chunks}"
    assert "Restarting process." in retry_messages[0]["content"]


def test_exhausting_the_retries_gives_up_with_a_clear_message():
    """Permanent failure ends the run with a message, not a hang or a silent drop.

    The bound is what makes this terminate. Without it a process that can never
    be written to would spin restarting forever and the caller would wait
    indefinitely.
    """
    lang = _with_process(_language(), write_errors=1)
    lang.restart_failures.extend(
        [[BrokenPipeError("still broken")] for _ in range(8)]
    )

    chunks = list(lang.run("echo hi"))

    final = [c for c in chunks if "Maximum retries reached" in str(c.get("content", ""))]
    assert final, f"expected a give-up message, got {chunks}"
    assert final[-1]["type"] == "console"
    assert final[-1]["format"] == "output"


def test_a_preprocessing_failure_is_reported_and_stops_the_run():
    """If preprocessing raises, the traceback is yielded and nothing is executed."""
    lang = _with_process(_language())
    lang.preprocess_code = mock.Mock(side_effect=ValueError("preprocess exploded"))

    chunks = list(lang.run("echo hi"))

    assert len(chunks) == 1
    assert chunks[0]["type"] == "console"
    assert chunks[0]["format"] == "output"
    assert "preprocess exploded" in chunks[0]["content"]
    lang.process.stdin.write.assert_not_called()
