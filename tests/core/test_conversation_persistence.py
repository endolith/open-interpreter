import json
import os
import tempfile
import unittest
from unittest import mock

from interpreter.core.core import OpenInterpreter


class TestConversationSurvivesAnAbandonedStream(unittest.TestCase):
    """The conversation must be on disk even if the reader walks away mid-turn.

    Ctrl-C surfaces as GeneratorExit at whatever point the stream happens to be
    suspended, so persistence cannot live on the normal completion path -- there
    is no normal completion. It lives in a `finally:` inside _streaming_chat,
    which writes to a temp file and os.replace()s it so an interrupt cannot leave
    a half-written conversation behind.

    None of that is covered by a test. A refactor that moves that `finally:` one
    indentation level, or drops the atomic replace for a plain write, loses
    conversations with nothing failing: the reader sees an ordinary short turn
    and the loss only shows up later, when a resumed conversation is missing its
    last few messages.
    """

    def _interpreter(self, tmp):
        # disable_telemetry is what anonymous_telemetry derives from; setting the
        # property directly is not possible (it is read-only).
        interpreter = OpenInterpreter(
            conversation_history_path=tmp, disable_telemetry=True
        )
        interpreter.conversation_history = True
        return interpreter

    def _stubbed_stream(self, interpreter, chunks):
        """Replace the respond loop so the test controls where it stops.

        _respond_and_store is the seam the rest of the suite patches; stubbing it
        keeps this test about persistence rather than about the LLM, the
        terminal, or chunk ordering.
        """
        def _fake():
            for chunk in chunks:
                # The real _respond_and_store appends to interpreter.messages as
                # it goes. Without that here, the saved conversation contains only
                # the user's message and every assertion about it passes
                # vacuously.
                interpreter.messages.append(chunk)
                yield chunk

        interpreter._respond_and_store = _fake
        return interpreter

    def _conversation_files(self, tmp):
        return [f for f in os.listdir(tmp) if f.endswith(".json")]

    def test_conversation_is_saved_when_the_stream_is_abandoned(self):
        """Breaking out of the stream early still persists the conversation."""
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = self._interpreter(tmp)
            self._stubbed_stream(
                interpreter,
                [
                    {"role": "assistant", "type": "message", "content": "first"},
                    {"role": "assistant", "type": "message", "content": "second"},
                ],
            )

            stream = interpreter._streaming_chat(message="hello", display=False)
            # Consume one chunk, then walk away the way Ctrl-C does. close() is
            # what throws GeneratorExit into the suspended generator.
            next(stream)
            stream.close()

            files = self._conversation_files(tmp)
            self.assertEqual(
                len(files), 1, f"expected exactly one saved conversation, got {files}"
            )
            with open(os.path.join(tmp, files[0]), encoding="utf-8") as f:
                saved = json.load(f)
            self.assertGreaterEqual(
                len(saved), 1, "the conversation file must not be empty"
            )

    def test_save_happens_even_if_the_respond_loop_raises(self):
        """An exception mid-turn still persists what was said so far.

        The failure and the abandonment are different shapes -- one unwinds
        through `except`, the other through GeneratorExit -- and only the second
        is a Ctrl-C. Both have to reach the same save.
        """
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = self._interpreter(tmp)

            def _boom():
                interpreter.messages.append(
                    {
                        "role": "assistant",
                        "type": "message",
                        "content": "partial",
                    }
                )
                yield {
                    "role": "assistant",
                    "type": "message",
                    "content": "partial",
                }
                raise RuntimeError("provider died mid-turn")

            interpreter._respond_and_store = _boom

            with self.assertRaises(RuntimeError):
                for _ in interpreter._streaming_chat(
                    message="hello", display=False
                ):
                    pass

            files = self._conversation_files(tmp)
            self.assertEqual(
                len(files), 1, f"a mid-turn error must still save, got {files}"
            )

    def test_no_partial_file_is_left_behind(self):
        """The temp file used for the atomic write does not survive.

        os.replace() is what makes the write atomic. If it were a plain write, an
        interrupt partway through would leave truncated JSON that fails to load
        on resume -- and the user's conversation is gone either way, just more
        confusingly. Also checks no dot-prefixed temp files are left in the
        directory.
        """
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = self._interpreter(tmp)
            self._stubbed_stream(
                interpreter,
                [{"role": "assistant", "type": "message", "content": "x"}],
            )
            for _ in interpreter._streaming_chat(message="hello", display=False):
                pass

            leftovers = [f for f in os.listdir(tmp) if f.startswith(".")]
            self.assertEqual(
                leftovers, [], f"temp files left behind: {leftovers}"
            )

    def test_saved_file_is_valid_json_containing_the_messages(self):
        """What lands on disk parses and holds the conversation, not a stub."""
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = self._interpreter(tmp)
            self._stubbed_stream(
                interpreter,
                [
                    {"role": "assistant", "type": "message", "content": "alpha"},
                    {"role": "assistant", "type": "message", "content": "beta"},
                ],
            )
            stream = interpreter._streaming_chat(message="hello", display=False)
            next(stream)
            next(stream)
            stream.close()

            files = self._conversation_files(tmp)
            with open(os.path.join(tmp, files[0]), encoding="utf-8") as f:
                saved = json.load(f)
            self.assertIsInstance(saved, list)
            self.assertTrue(saved, "saved conversation must not be empty")
            for entry in saved:
                self.assertIsInstance(entry, dict)
                self.assertIn("role", entry)

    def test_no_conversation_file_when_history_is_disabled(self):
        """conversation_history=False writes nothing.

        Guards the other side of the condition in the save, so a future change
        cannot make persistence unconditional and start writing files for users
        who turned it off.
        """
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = OpenInterpreter(
                conversation_history_path=tmp, disable_telemetry=True
            )
            interpreter.conversation_history = False
            self._stubbed_stream(
                interpreter,
                [{"role": "assistant", "type": "message", "content": "x"}],
            )
            for _ in interpreter._streaming_chat(message="hello", display=False):
                pass
            self.assertEqual(self._conversation_files(tmp), [])

    def test_save_goes_through_a_single_atomic_rename(self):
        """The destination is written by one os.replace, never incrementally.

        Atomicity cannot be observed from the outside: replacing the rename with
        a plain copy still leaves a complete, valid file afterwards, and the
        difference only shows when the process dies *between* the write and the
        move. That timing gap is not unit-testable without a real kill, so the
        mechanism is pinned directly instead -- os.replace is called exactly once,
        with a temp file in the same directory and the final conversation file as
        the destination. A copy or a second write shows up here immediately.
        """
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = self._interpreter(tmp)
            self._stubbed_stream(
                interpreter,
                [{"role": "assistant", "type": "message", "content": "x"}],
            )
            calls = []
            real_replace = os.replace

            def _record(src, dst):
                calls.append((src, dst))
                return real_replace(src, dst)

            with mock.patch("interpreter.core.core.os.replace", _record):
                for _ in interpreter._streaming_chat(
                    message="hello", display=False
                ):
                    pass

            self.assertEqual(
                len(calls), 1, f"expected one atomic rename, got {calls}"
            )
            src, dst = calls[0]
            self.assertNotEqual(src, dst, "the write must not target the destination")
            self.assertEqual(
                os.path.dirname(os.path.abspath(src)),
                os.path.abspath(tmp),
                "the temp file must be on the same filesystem as the destination, "
                "or the rename is not atomic",
            )
            self.assertEqual(
                os.path.basename(dst),
                interpreter.conversation_filename,
            )

    def test_persistence_survives_a_failing_json_dump(self):
        """A dump failure cleans up its temp file and does not mask the error.

        The save runs inside the same `finally:` as the abandoned stream, so an
        exception raised *by the save* would replace whatever was propagating.
        The inner `finally:` unlinks the temp file, so the directory is left as it
        was found rather than accumulating .tmp files.
        """
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = self._interpreter(tmp)

            class Unserialisable:
                pass

            self._stubbed_stream(
                interpreter,
                [
                    {
                        "role": "assistant",
                        "type": "message",
                        "content": Unserialisable(),
                    }
                ],
            )
            with self.assertRaises(TypeError):
                for _ in interpreter._streaming_chat(
                    message="hello", display=False
                ):
                    pass

            leftovers = [f for f in os.listdir(tmp) if f.startswith(".")]
            self.assertEqual(
                leftovers, [], f"temp files left after a failed dump: {leftovers}"
            )


if __name__ == "__main__":
    unittest.main()