"""Characterization tests for ``computer.ai`` text chunking and chat helpers.

LLM calls are mocked; no real model or API is touched.
"""

from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.ai import ai as ai_mod
from interpreter.core.computer.ai.ai import Ai, chunk_responses, fast_llm, split_into_chunks


def _fake_encoding():
    """A fake tiktoken encoding where each character counts as one token."""

    return SimpleNamespace(
        encode=lambda text: list(range(len(text))),
        decode=lambda tokens: "".join(chr(97 + i) for i in tokens),
    )


def test_split_into_chunks_uses_tokenizer():
    """split_into_chunks() splits on the model's tokenizer when available."""
    llm = SimpleNamespace(model="gpt-4")
    with mock.patch.object(
        ai_mod.tiktoken, "encoding_for_model", return_value=_fake_encoding()
    ) as encoding_for_model:
        chunks = split_into_chunks("abcd", tokens=3, llm=llm, overlap=1)

    encoding_for_model.assert_called_once_with("gpt-4")
    assert chunks == ["abc", "cd"]


def test_split_into_chunks_falls_back_to_characters():
    """split_into_chunks() falls back to character slicing when tiktoken fails."""
    llm = SimpleNamespace(model="gpt-4")
    with mock.patch.object(
        ai_mod.tiktoken, "encoding_for_model", side_effect=Exception("no tiktoken")
    ):
        chunks = split_into_chunks("abcdefghij", tokens=2, llm=llm, overlap=0)

    assert chunks == ["abcdefgh", "ij"]


def test_chunk_responses_groups_by_token_limit():
    """chunk_responses() packs responses into chunks under the token budget."""
    llm = SimpleNamespace(model="gpt-4")
    with mock.patch.object(
        ai_mod.tiktoken, "encoding_for_model", return_value=_fake_encoding()
    ):
        chunks = chunk_responses(["ab", "c"], tokens=3, llm=llm)

    assert chunks == ["ab\n\nc"]


def test_chunk_responses_falls_back_to_characters():
    """chunk_responses() falls back to character budgets when tiktoken fails."""
    llm = SimpleNamespace(model="gpt-4")
    with mock.patch.object(
        ai_mod.tiktoken, "encoding_for_model", side_effect=Exception("no tiktoken")
    ):
        chunks = chunk_responses(["aaa", "bbbbbbbbbb", "cc"], tokens=2, llm=llm)

    assert chunks == ["bbbbbbbbbb", "aaa\n\ncc"]


def test_fast_llm_restores_interpreter_state():
    """fast_llm() swaps the interpreter's messages/system message for the call
    and restores them even on success."""
    interpreter = SimpleNamespace(
        messages=["old"], system_message="old_sys", chat=lambda m: [{"content": "answer"}]
    )
    llm = SimpleNamespace(interpreter=interpreter)

    result = fast_llm(llm, "sys", "user")

    assert result == "answer"
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
    assert messages[0] == {
        "role": "system",
        "type": "message",
        "content": "You are a helpful AI assistant.",
    }
    assert messages[1] == {"role": "user", "type": "message", "content": "hi"}


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
    assert messages[-1] == {
        "role": "user",
        "type": "image",
        "format": "base64",
        "content": "abc123",
    }
