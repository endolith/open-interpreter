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
