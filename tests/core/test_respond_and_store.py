from unittest import mock

from interpreter import OpenInterpreter


def test_respond_and_store_merges_consecutive_chunks():
    interpreter = OpenInterpreter()
    interpreter.messages = []

    def fake_respond(_interpreter):
        yield {"role": "assistant", "type": "message", "content": "Hello "}
        yield {"role": "assistant", "type": "message", "content": "world"}

    with mock.patch("interpreter.core.core.respond", fake_respond):
        chunks = list(interpreter._respond_and_store())

    assert interpreter.messages[-1]["content"] == "Hello world"
    assert any(c.get("start") for c in chunks)


def test_respond_and_store_skips_ephemeral_review_chunks():
    interpreter = OpenInterpreter()
    interpreter.messages = []

    def fake_respond(_interpreter):
        yield {"role": "assistant", "type": "review", "format": "safe", "content": "ok"}
        yield {"role": "assistant", "type": "message", "content": "done"}

    with mock.patch("interpreter.core.core.respond", fake_respond):
        list(interpreter._respond_and_store())

    assert len(interpreter.messages) == 1


def test_respond_and_store_truncates_console_output():
    interpreter = OpenInterpreter()
    interpreter.max_output = 100
    interpreter.messages = []

    def fake_respond(_interpreter):
        yield {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": "x" * 500,
        }

    with mock.patch("interpreter.core.core.respond", fake_respond):
        list(interpreter._respond_and_store())

    assert "Output truncated" in interpreter.messages[-1]["content"]
