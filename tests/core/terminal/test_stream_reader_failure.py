import threading
import unittest
from unittest import mock

from interpreter.core.terminal.languages.bash import Bash


class _ExplodingStream:
    """Yields a couple of real lines, then raises from readline().

    Stands in for the failure this guards: an unexpected exception raised from
    inside the reader's per-line loop, after some output has already been
    produced.
    """

    def __init__(self, lines, boom):
        self._lines = list(lines)
        self._boom = boom

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        raise self._boom


class TestStreamReaderSurvivesUnexpectedErrors(unittest.TestCase):
    """A reader failure must end the turn, not hang it.

    `done` is set in exactly two places, both inside the reader: the
    end-of-execution marker and the KeyboardInterrupt branch. If the reader dies
    from anything else, `done` is never set, and run()'s consumer loop -- a bare
    `while True` that only breaks on that event -- spins forever. The turn never
    ends, output already in the pipe is never read, and the Rich Live display
    stays up, so Ctrl-C arrives as a keystroke and appears to do nothing.

    The reported symptom was an output stream that "got swallowed" together with
    a terminal that could not be interrupted, which reads as two unrelated
    problems and is actually one.
    """

    def _reader(self):
        return Bash()

    def test_reader_sets_done_when_the_loop_raises(self):
        """An exception mid-stream still ends the turn.

        Asserted directly on the reader rather than through run(), so a
        regression fails immediately instead of hanging the suite.
        """
        reader = self._reader()
        stream = _ExplodingStream(["first line\n"], RuntimeError("boom"))

        reader.handle_stream_output(stream, False)

        self.assertTrue(
            reader.done.is_set(),
            "an unexpected reader failure must end the turn, or run() spins forever",
        )
        self.assertIsInstance(reader._stream_error, RuntimeError)

    def test_output_produced_before_the_failure_is_kept(self):
        """Lines read before the failure are still delivered.

        Aborting must not also discard what was already queued -- that would
        trade a hang for the silent loss the abort was meant to stop.
        """
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
        """The abort says itself in the output.

        A reader that dies quietly is indistinguishable from a command that
        printed nothing, which is how the original report ended up blaming the
        model for losing interest.
        """
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
        """End to end: run() returns instead of looping forever.

        The failure is injected one level down, by making the per-line hook the
        reader calls raise. Patching handle_stream_output itself would replace
        the method that contains the handling, so the test would pass or fail
        for reasons unrelated to the fix.

        Guarded by a watchdog so a regression is a failed assertion rather than a
        hung test run -- the consumer loop is `while True`, and this is the one
        test that would otherwise never come back.
        """
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
            Bash, "detect_active_line", side_effect=RuntimeError("boom")
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


if __name__ == "__main__":
    unittest.main()