"""Characterization tests for ``computer.ai`` chat helpers.

LLM calls are mocked; no real model or API is touched. ``split_into_chunks``
and ``chunk_responses`` are covered by ``test_ai_helpers.py`` (the pre-existing
tests on main).
"""

from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.ai import ai as ai_mod
from interpreter.core.computer.ai.ai import Ai, fast_llm


def test_fast_llm_restores_interpreter_state():
    """fast_llm() swaps the interpreter's messages/system message for the call
    and restores them even on success."""
    seen = {}

    def chat(message):
        seen["message"] = message
        seen["messages"] = interpreter.messages
        seen["system_message"] = interpreter.system_message
        return [{"content": "answer"}]

    interpreter = SimpleNamespace(messages=["old"], system_message="old_sys", chat=chat)
    llm = SimpleNamespace(interpreter=interpreter)

    result = fast_llm(llm, "sys", "user")

    assert result == "answer"
    assert seen["message"] == "user"
    assert seen["messages"] == []
    assert seen["system_message"] == "sys"
    assert interpreter.messages == ["old"]
    assert interpreter.system_message == "old_sys"


def test_query_map_chunks_queries_each_chunk():
    """query_map_chunks() runs fast_llm over every chunk and returns the
    responses in the order the chunks were given."""
    llm = SimpleNamespace()

    def fake_fast_llm(llm, query, chunk):
        return f"response:{chunk}"

    with mock.patch.object(ai_mod, "fast_llm", side_effect=fake_fast_llm) as fast:
        responses = ai_mod.query_map_chunks(["a", "b"], llm, "q")

    assert responses == ["response:a", "response:b"]
    fast.assert_any_call(llm, "q", "a")
    fast.assert_any_call(llm, "q", "b")


def test_ai_chat_concatenates_llm_output():
    """Ai.chat() sends a system + user message to llm.run and concatenates the
    content chunks."""
    computer = SimpleNamespace(
        interpreter=SimpleNamespace(
            llm=SimpleNamespace(
                run=mock.Mock(return_value=[{"content": "hello"}, {"content": " world"}])
            )
        )
    )
    ai = Ai(computer)

    result = ai.chat("hi")

    assert result == "hello world"
    messages = computer.interpreter.llm.run.call_args[0][0]
    assert messages == [
        {
            "role": "system",
            "type": "message",
            "content": "You are a helpful AI assistant.",
        },
        {"role": "user", "type": "message", "content": "hi"},
    ]


def test_ai_chat_appends_image_message_for_base64():
    """Ai.chat(base64=...) adds an image message after the user message."""
    computer = SimpleNamespace(
        interpreter=SimpleNamespace(
            llm=SimpleNamespace(run=mock.Mock(return_value=[]))
        )
    )
    ai = Ai(computer)

    ai.chat("hi", base64="abc123")

    messages = computer.interpreter.llm.run.call_args[0][0]
    assert messages == [
        {
            "role": "system",
            "type": "message",
            "content": "You are a helpful AI assistant.",
        },
        {"role": "user", "type": "message", "content": "hi"},
        {"role": "user", "type": "image", "format": "base64", "content": "abc123"},
    ]


def test_query_reduce_chunks_returns_single_response_unchanged():
    """A single mapped response is returned as-is, with no reduce call.

    The while loop's body never runs for one response, so the old code fell
    through to a `summaries` that was never assigned and raised
    UnboundLocalError. Any text short enough to produce one chunk took this
    path, which made computer.ai.summarize() fail on short input (#209).
    """
    llm = SimpleNamespace()

    with mock.patch.object(ai_mod, "fast_llm") as fast:
        assert ai_mod.query_reduce_chunks(["only"], llm, 2000, "q") == "only"

    fast.assert_not_called()


def test_query_reduce_chunks_returns_none_for_no_responses():
    """No responses reduces to None rather than raising."""
    llm = SimpleNamespace()

    with mock.patch.object(ai_mod, "fast_llm") as fast:
        assert ai_mod.query_reduce_chunks([], llm, 2000, "q") is None

    fast.assert_not_called()


def test_query_reduce_chunks_reduces_until_one_response_remains():
    """Each pass feeds its summaries into the next, so the list converges to one.

    The old loop never reassigned `responses`, so `len(responses) > 1` stayed
    true forever and it made unbounded LLM calls for any input that mapped to
    two or more responses.
    """
    llm = SimpleNamespace()
    calls = []

    def fake_fast_llm(llm, query, chunk):
        calls.append(chunk)
        return f"summary({chunk})"

    def fake_chunk_responses(responses, tokens, llm):
        # Halve the list each pass by pairing neighbours.
        return [
            "+".join(responses[i : i + 2]) for i in range(0, len(responses), 2)
        ]

    with mock.patch.object(ai_mod, "fast_llm", side_effect=fake_fast_llm), \
         mock.patch.object(ai_mod, "chunk_responses", side_effect=fake_chunk_responses):
        result = ai_mod.query_reduce_chunks(["a", "b", "c", "d"], llm, 2000, "q")

    assert result == "summary(summary(a+b)+summary(c+d))"
    assert calls == ["a+b", "c+d", "summary(a+b)+summary(c+d)"]


def test_query_reduce_chunks_stops_when_a_pass_makes_no_progress():
    """Responses that each fill a chunk alone are merged in one final call.

    chunk_responses keeps an oversized response as its own chunk, so a pass
    can return as many summaries as it was given. Without a guard that is an
    unbounded loop of paid LLM calls at constant length.
    """
    llm = SimpleNamespace()
    calls = []

    def fake_fast_llm(llm, query, chunk):
        calls.append(chunk)
        return "same-size-summary"

    with mock.patch.object(ai_mod, "fast_llm", side_effect=fake_fast_llm), \
         mock.patch.object(ai_mod, "chunk_responses", side_effect=lambda r, t, l: list(r)):
        result = ai_mod.query_reduce_chunks(["big1", "big2"], llm, 2000, "q")

    assert result == "same-size-summary"
    # One pass over both responses, then a single merge of what was left.
    assert calls == ["big1", "big2", "same-size-summary\n\nsame-size-summary"]
