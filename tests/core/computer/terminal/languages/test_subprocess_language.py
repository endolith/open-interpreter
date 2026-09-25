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
