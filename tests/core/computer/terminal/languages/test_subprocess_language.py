import sys
from io import StringIO
from unittest import mock

import pytest

from interpreter.core.computer.terminal.languages.subprocess_language import (
    SubprocessLanguage,
)


class EchoLanguage(SubprocessLanguage):
    file_extension = "txt"
    name = "Echo"

    def __init__(self):
        super().__init__()
        self.start_cmd = ["cat"]

    def detect_end_of_execution(self, line):
        return "##done##" in line


def test_handle_stream_output_puts_console_chunks():
    """handle_stream_output() enqueues console output and signals completion at end markers."""
    lang = EchoLanguage()
    stream = StringIO("hello\n##done##\n")
    lang.handle_stream_output(stream, is_error_stream=False)
    assert lang.output_queue.get()["content"] == "hello\n"
    assert lang.done.is_set()


def test_handle_stream_output_active_line():
    """handle_stream_output() emits active_line chunks when detect_active_line matches."""
    lang = EchoLanguage()

    def detect(line):
        if "##active_line2##" in line:
            return 2
        return None

    lang.detect_active_line = detect
    stream = StringIO("prefix ##active_line2## suffix\n")
    lang.handle_stream_output(stream, is_error_stream=False)
    active = lang.output_queue.get()
    assert active["format"] == "active_line"
    assert active["content"] == 2
    output = lang.output_queue.get()
    assert "suffix" in output["content"]


def test_handle_stream_output_keyboard_interrupt_on_stderr():
    """handle_stream_output() treats KeyboardInterrupt on stderr as output and ends execution."""
    lang = EchoLanguage()
    stream = StringIO("KeyboardInterrupt\n")
    lang.handle_stream_output(stream, is_error_stream=True)
    assert lang.output_queue.get()["content"] == "KeyboardInterrupt"
    assert lang.done.is_set()


def test_run_yields_queue_output():
    """SubprocessLanguage.run() writes code to stdin and yields queued console chunks."""
    lang = EchoLanguage()
    mock_process = mock.Mock()
    mock_process.stdin = mock.Mock()
    mock_process.stdout = StringIO("")
    mock_process.stderr = StringIO("")
    lang.process = mock_process

    lang.output_queue.put(
        {"type": "console", "format": "output", "content": "result"}
    )
    lang.done.set()

    with mock.patch.object(lang, "start_process"):
        with mock.patch("interpreter.core.computer.terminal.languages.subprocess_language.time.sleep"):
            chunks = []
            for chunk in lang.run("echo hi"):
                chunks.append(chunk)
                break
    mock_process.stdin.write.assert_called_once_with("echo hi\n")
    mock_process.stdin.flush.assert_called_once()
    assert chunks[0]["content"] == "result"


def test_handle_stream_output_plain_chunk_dict_is_exact():
    """A plain output line becomes an exact console/output chunk.

    The chunk dict is the pipeline's currency; asserting only "content" let
    every key rename and format/value change survive, so the whole dict is
    pinned here.
    """
    lang = EchoLanguage()
    stream = StringIO("hello\n##done##\n")

    lang.handle_stream_output(stream, is_error_stream=False)

    assert lang.output_queue.get() == {
        "type": "console",
        "format": "output",
        "content": "hello\n",
    }


def test_handle_stream_output_active_line_chunk_dict_is_exact():
    """The active-line chunk is exactly a console/active_line chunk with the int.

    The UI switches on format=="active_line" and numeric content; a renamed key
    or a string content would break line highlighting.
    """
    lang = EchoLanguage()

    def detect(line):
        return 2 if "##active_line2##" in line else None

    lang.detect_active_line = detect
    stream = StringIO("##active_line2##\n")

    lang.handle_stream_output(stream, is_error_stream=False)

    assert lang.output_queue.get() == {
        "type": "console",
        "format": "active_line",
        "content": 2,
    }


def test_active_line_marker_is_stripped_from_trailing_output():
    """After an active-line marker, the marker regex is removed from the remainder.

    The leftover text is sent as output; the marker itself must not leak into it.
    """
    lang = EchoLanguage()

    def detect(line):
        return 2 if "##active_line2##" in line else None

    lang.detect_active_line = detect
    stream = StringIO("##active_line2## leftover\n")

    lang.handle_stream_output(stream, is_error_stream=False)

    lang.output_queue.get()  # active_line chunk
    trailing = lang.output_queue.get()
    assert trailing == {
        "type": "console",
        "format": "output",
        "content": " leftover\n",
    }
    assert "##active_line" not in trailing["content"]


def test_end_of_execution_marker_is_stripped_but_trailing_text_kept():
    """The end marker is removed from the line and any trailing text is emitted.

    A mutated marker literal would leave the marker in the output, and the
    trailing text still has to be sent before done is set.
    """
    lang = EchoLanguage()
    # Use the real end marker literal so the strip branch fires, plus a detector
    # that recognises it for this echo language.
    lang.detect_end_of_execution = lambda line: "##end_of_execution##" in line
    stream = StringIO("done ##end_of_execution##\n")

    lang.handle_stream_output(stream, is_error_stream=False)

    assert lang.output_queue.get() == {
        "type": "console",
        "format": "output",
        "content": "done",
    }
    assert lang.done.is_set()


def test_keyboard_interrupt_chunk_dict_is_exact():
    """A stderr KeyboardInterrupt emits an exact console chunk and ends the run."""
    lang = EchoLanguage()
    stream = StringIO("KeyboardInterrupt\n")

    lang.handle_stream_output(stream, is_error_stream=True)

    assert lang.output_queue.get() == {
        "type": "console",
        "format": "output",
        "content": "KeyboardInterrupt",
    }
    assert lang.done.is_set()


def test_none_from_postprocessor_discards_the_line():
    """A postprocessor returning None discards that line entirely.

    The None sentinel is how languages filter their own noise; if the check
    broke, the None would be queued as an output chunk.
    """
    lang = EchoLanguage()
    lang.line_postprocessor = lambda line: None

    import queue as _queue

    stream = StringIO("noise\n##done##\n")
    lang.handle_stream_output(stream, is_error_stream=False)

    assert lang.output_queue.empty()


class DyingLanguage(SubprocessLanguage):
    """A language whose process exits instead of printing the end marker.

    Stands in for a shell block that runs `exit` or hits a fatal syntax error:
    the process is alive when the code is written to stdin and gone before
    ##done## could be printed.
    """

    file_extension = "txt"
    name = "Dying"

    def __init__(self):
        super().__init__()
        self.start_cmd = [
            sys.executable,
            "-c",
            "import sys; sys.stdin.readline(); print('bye', flush=True); sys.exit(3)",
        ]

    def detect_end_of_execution(self, line):
        return "##done##" in line


@pytest.mark.timeout(30)
def test_run_returns_when_the_process_dies_before_the_end_marker():
    """run() ends, reports the exit code, and drops the dead process.

    The end-of-execution marker is the only completion signal, so a process
    that exits before printing it (a shell block calling `exit`, or one that
    bash refuses to parse) used to leave run() spinning forever with no
    output, no error and no prompt. The output it did produce must still be
    yielded, and the next run() must get a fresh process.
    """
    lang = DyingLanguage()
    chunks = list(lang.run("anything"))
    output = "".join(
        chunk["content"] for chunk in chunks if chunk.get("format") == "output"
    )
    assert "bye" in output
    assert "exited" in output
    assert "3" in output
    assert lang.process is None


def test_handle_stream_output_survives_a_raising_detect_active_line():
    """A language whose marker parse raises does not lose its reader thread.

    Every subprocess language parses the active-line marker itself, and most
    do it with an unguarded int() over anything containing "##active_line".
    Program output is untrusted, so a line that merely resembles a marker
    makes that parse raise. If the exception escapes here the reader thread
    dies, nothing drains the pipe, the end marker is never seen, and run()
    waits forever — on this block and on every later one using the same
    process. The line is kept as ordinary output instead.
    """
    lang = EchoLanguage()

    def detect(line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    lang.detect_active_line = detect
    stream = StringIO("echoed ##active_line## not a number\n")

    lang.handle_stream_output(stream, is_error_stream=False)

    output = lang.output_queue.get_nowait()
    assert output["format"] == "output"
    assert "not a number" in output["content"]
    assert lang.output_queue.empty()


def test_handle_stream_output_calls_detect_active_line_once_per_line():
    """detect_active_line() runs once per line rather than twice.

    It was called once as a truthiness test and again for its value, so any
    per-line cost was paid twice and a marker numbered 0 would be discarded
    by the first call and re-parsed as plain output by the second.
    """
    lang = EchoLanguage()
    calls = []

    def detect(line):
        calls.append(line)
        return 7 if "##active_line7##" in line else None

    lang.detect_active_line = detect
    stream = StringIO("##active_line7##\n")

    lang.handle_stream_output(stream, is_error_stream=False)

    assert len(calls) == 1
    assert lang.output_queue.get_nowait()["content"] == 7


def test_join_stream_threads_reports_an_unfinished_reader():
    """A timed reader join must not be reported as a complete drain.

    If a reader is still alive when the process exits, run() cannot prove the
    queue is complete; it must surface that instead of silently dropping output.
    """
    lang = EchoLanguage()

    class Reader:
        def __init__(self, alive):
            self.alive = alive

        def join(self, timeout=None):
            self.timeout = timeout

        def is_alive(self):
            return self.alive

    lang.stream_threads = [Reader(False), Reader(True)]

    assert lang._join_stream_threads() is False
    assert lang.stream_threads[0].timeout == 5


def test_run_warns_when_a_reader_is_still_alive_after_exit():
    """run() must not present a possibly incomplete drain as complete."""
    lang = EchoLanguage()
    lang.process = mock.Mock()
    lang.process.poll.return_value = 1

    class Reader:
        def join(self, timeout=None):
            self.timeout = timeout

        def is_alive(self):
            return True

    lang.stream_threads = [Reader()]

    chunks = list(lang.run("anything"))
    output = "\n".join(str(chunk.get("content")) for chunk in chunks)

    assert "output reader was still running" in output
    assert lang.process is None
