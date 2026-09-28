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
