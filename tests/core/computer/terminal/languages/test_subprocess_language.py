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
