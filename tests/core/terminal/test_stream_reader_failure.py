"""A reader failure must end the turn, not hang it.

``done`` is set in exactly two places, both inside the reader: the
end-of-execution marker and the KeyboardInterrupt branch. If the reader dies
from anything else, ``done`` is never set, and ``run()``'s consumer loop -- a
bare ``while True`` that only breaks on that event -- spins forever. The turn
never ends, output already in the pipe is never read, and the Rich Live display
stays up, so Ctrl-C arrives as a keystroke and appears to do nothing.

Ported from classic/develop (871fdca4fe).
"""

import threading
import unittest
from unittest import mock

from interpreter.core.computer.terminal.languages.shell import Shell


class _ExplodingStream:
    """Yields a couple of real lines, then raises from readline()."""

    def __init__(self, lines, boom):
        self._lines = list(lines)
        self._boom = boom

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        raise self._boom


class TestStreamReaderSurvivesUnexpectedErrors(unittest.TestCase):
    """A reader failure must end the turn, not hang it."""

    def _reader(self):
        return Shell()

    def test_reader_sets_done_when_the_loop_raises(self):
        """An exception mid-stream still ends the turn."""
        reader = self._reader()
        stream = _ExplodingStream(["first line\n"], RuntimeError("boom"))

        reader.handle_stream_output(stream, False)

        self.assertTrue(
            reader.done.is_set(),
            "an unexpected reader failure must end the turn, or run() spins forever",
        )
        self.assertIsInstance(reader._stream_error, RuntimeError)

    def test_output_produced_before_the_failure_is_kept(self):
        """Lines read before the failure are still delivered."""
        reader = self._reader()
        stream = _ExplodingStream(
            ["kept line\n", "##end_of_execution##\n"], RuntimeError("boom")
        )

        reader.handle_stream_output(stream, False)

        seen = []
        while not reader.output_queue.empty():
            seen.append(reader.output_queue.get_nowait())
        self.assertTrue(
            any("kept line" in str(chunk.get("content")) for chunk in seen),
            f"output read before the failure was dropped: {seen}",
        )

    def test_failure_is_reported_rather_than_silent(self):
        """The abort says itself in the output."""
        reader = self._reader()
        stream = _ExplodingStream([], RuntimeError("boom"))

        reader.handle_stream_output(stream, False)

        seen = []
        while not reader.output_queue.empty():
            seen.append(str(reader.output_queue.get_nowait().get("content")))
        joined = "\n".join(seen)
        self.assertIn("output stream stopped unexpectedly", joined)
        self.assertIn("boom", joined)

    def test_run_terminates_when_the_loop_raises(self):
        """End to end: run() returns instead of looping forever."""
        reader = self._reader()
        collected = []
        finished = threading.Event()

        def _drain():
            try:
                for chunk in reader.run("echo hello"):
                    collected.append(chunk)
            finally:
                finished.set()

        with mock.patch.object(
            Shell, "detect_active_line", side_effect=RuntimeError("boom")
        ):
            worker = threading.Thread(target=_drain, daemon=True)
            worker.start()
            try:
                self.assertTrue(
                    finished.wait(timeout=20),
                    f"run() did not terminate; collected {collected!r}",
                )
            finally:
                reader.terminate()

        joined = "\n".join(str(chunk.get("content")) for chunk in collected)
        self.assertIn("output stream stopped unexpectedly", joined)

    def test_reader_failure_discards_the_process(self):
        """The dead process must not be reusable by the next run()."""
        reader = self._reader()
        process = mock.Mock()
        reader.process = process

        reader.handle_stream_output(_ExplodingStream([], RuntimeError("boom")), False)

        process.terminate.assert_called_once()
        self.assertIsNone(reader.process)

    def test_poll_process_handles_a_cleared_handle(self):
        """The liveness check returns None after a reader abort clears the handle."""
        reader = self._reader()
        reader.process = None
        assert reader._poll_process() is None

        process = mock.Mock()
        process.poll.return_value = 3
        reader.process = process
        assert reader._poll_process() == 3

    def test_exit_check_tolerates_a_process_cleared_by_abort(self):
        """run()'s liveness check must not crash if the reader cleared the process."""
        reader = self._reader()
        process = mock.Mock()
        reader.process = process
        reader.handle_stream_output(_ExplodingStream([], RuntimeError("boom")), False)

        assert reader.process is None
        assert reader.done.is_set()
