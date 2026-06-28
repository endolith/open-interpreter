import queue
from io import StringIO
from unittest import mock

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
    lang = EchoLanguage()
    stream = StringIO("hello\n##done##\n")
    lang.handle_stream_output(stream, is_error_stream=False)
    assert lang.output_queue.get()["content"] == "hello\n"
    assert lang.done.is_set()


def test_handle_stream_output_active_line():
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
    lang = EchoLanguage()
    stream = StringIO("KeyboardInterrupt\n")
    lang.handle_stream_output(stream, is_error_stream=True)
    assert lang.output_queue.get()["content"] == "KeyboardInterrupt"
    assert lang.done.is_set()


def test_run_yields_queue_output():
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
