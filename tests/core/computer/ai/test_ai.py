"""Characterization tests for ``computer.ai`` chat helpers.

LLM calls are mocked; no real model or API is touched. ``split_into_chunks``
and ``chunk_responses`` are covered by ``test_ai_helpers.py`` (the pre-existing
tests on main).
"""

from types import SimpleNamespace
import pytest
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


@pytest.mark.xfail(
    raises=AssertionError,
    reason="#389: the loop never reassigns `responses`, so it never terminates",
)
def test_query_reduce_chunks_collapses_many_responses_to_one():
    """The reduce loop folds many responses into one summary.

    Intended behaviour, currently impossible: `responses` is read by
    `while len(responses) > 1` and never written, so the condition cannot change.
    See #389. The stub raises after a few rounds so a broken loop fails instead of
    hanging the suite, which is the whole failure mode.
    """
    from interpreter.core.computer.ai.ai import query_reduce_chunks

    rounds = []

    def fake_chunk_responses(responses, chunk_size, llm):
        rounds.append(list(responses))
        if len(rounds) > 3:
            raise AssertionError("reduce loop did not terminate")
        return ["joined"]

    llm = mock.Mock()
    with mock.patch(
        "interpreter.core.computer.ai.ai.chunk_responses",
        side_effect=fake_chunk_responses,
    ):
        with mock.patch(
            "interpreter.core.computer.ai.ai.fast_llm", return_value="the summary"
        ):
            result = query_reduce_chunks(["a", "b", "c"], llm, 10, "summarise")

    assert result == "the summary"
    assert rounds == [["a", "b", "c"]]


@pytest.mark.xfail(
    raises=UnboundLocalError,
    reason="#389: `summaries` is unbound when the loop body never runs",
)
def test_query_reduce_chunks_returns_a_single_response_unchanged():
    """One response is already a summary and should come back untouched.

    Currently raises UnboundLocalError, because `return summaries[0]` reads a
    name only assigned inside the loop. See #389.
    """
    from interpreter.core.computer.ai.ai import query_reduce_chunks

    llm = mock.Mock()
    result = query_reduce_chunks(["only"], llm, 10, "summarise")

    assert result == "only"


def _bounded_fast_llm(calls, limit=3, result="s"):
    """A fast_llm stub that records its calls then raises to break the loop.

    `query_reduce_chunks` never shrinks `responses` (#389), so any test that
    exercises one iteration will otherwise loop forever. Raising out of the stub
    is the only way to observe a single iteration without a timeout.
    """

    def _inner(llm, query, chunk):
        calls.append((llm, query, chunk))
        if len(calls) >= limit:
            raise RuntimeError("stop: loop is unbounded")
        return result

    return _inner


def test_query_reduce_chunks_summarises_each_chunk_separately():
    """Every chunk the chunker produces gets its own summary call.

    The reduction is parallel-per-chunk, so dropping one loses content that was
    present in the input.
    """
    from interpreter.core.computer.ai.ai import query_reduce_chunks

    calls = []
    llm = mock.Mock()
    with mock.patch(
        "interpreter.core.computer.ai.ai.chunk_responses", return_value=["one", "two"]
    ):
        with mock.patch(
            "interpreter.core.computer.ai.ai.fast_llm",
            side_effect=_bounded_fast_llm(calls, limit=2),
        ):
            with pytest.raises(RuntimeError):
                query_reduce_chunks(["a", "b"], llm, 10, "summarise")

    assert [c[2] for c in calls] == ["one", "two"]


def test_query_reduce_chunks_passes_the_query_to_every_summary():
    """The query is forwarded to each fast_llm call unchanged.

    The summary prompt is the caller's only way to steer the reduction, so a
    dropped or reordered argument silently produces unrelated summaries.
    """
    from interpreter.core.computer.ai.ai import query_reduce_chunks

    calls = []
    llm = mock.Mock()
    with mock.patch(
        "interpreter.core.computer.ai.ai.chunk_responses", return_value=["one", "two"]
    ):
        with mock.patch(
            "interpreter.core.computer.ai.ai.fast_llm",
            side_effect=_bounded_fast_llm(calls, limit=2),
        ):
            with pytest.raises(RuntimeError):
                query_reduce_chunks(["a", "b"], llm, 42, "what happened?")

    assert len(calls) == 2
    for llm_arg, query, chunk in calls:
        assert llm_arg is llm
        assert query == "what happened?"
        assert chunk in ("one", "two")


def test_query_splits_then_maps_then_reduces():
    """query() runs the map-then-reduce pipeline in order.

    Each stage is a separate unit that can fail on its own: if the split produces
    no chunks, the map never runs; if reduce is skipped, the raw per-chunk
    answers are returned as if they were a single answer.
    """
    computer = SimpleNamespace(
        interpreter=SimpleNamespace(llm=SimpleNamespace(model="gpt-4"))
    )
    ai = Ai(computer)

    with mock.patch(
        "interpreter.core.computer.ai.ai.split_into_chunks", return_value=["c1", "c2"]
    ) as split:
        with mock.patch(
            "interpreter.core.computer.ai.ai.query_map_chunks", return_value=["r1", "r2"]
        ) as mapper:
            with mock.patch(
                "interpreter.core.computer.ai.ai.query_reduce_chunks", return_value="final"
            ) as reducer:
                result = ai.query("body", "my query")

    assert result == "final"
    # split_into_chunks is called positionally: (text, tokens, llm, overlap).
    assert split.call_args.args[1] == 2000
    assert split.call_args.args[3] == 50
    assert mapper.call_args.args[0] == ["c1", "c2"]
    assert reducer.call_args.args[0] == ["r1", "r2"]


def test_query_defaults_the_reduce_prompt_to_the_map_prompt():
    """With no custom reduce prompt, the same question is used for both stages.

    Reducing with a different question than was asked silently changes the
    answer, so the default is a real contract rather than a convenience.
    """
    computer = SimpleNamespace(
        interpreter=SimpleNamespace(llm=SimpleNamespace(model="gpt-4"))
    )
    ai = Ai(computer)

    with mock.patch(
        "interpreter.core.computer.ai.ai.split_into_chunks", return_value=["c"]
    ):
        with mock.patch(
            "interpreter.core.computer.ai.ai.query_map_chunks", return_value=["r"]
        ):
            with mock.patch(
                "interpreter.core.computer.ai.ai.query_reduce_chunks", return_value="final"
            ) as reducer:
                ai.query("body", "same question")

    # query_reduce_chunks(responses, llm, chunk_size, custom_reduce_query)
    assert reducer.call_args.args[3] == "same question"


def test_summarize_uses_a_different_prompt_for_reducing_than_for_mapping():
    """summarize() maps with the summarisation prompt and reduces with a merge prompt.

    The two prompts are deliberately different: the second says to merge
    already-summarised texts. Passing the first to reduce would ask the model to
    re-summarise a summary.
    """
    computer = SimpleNamespace(
        interpreter=SimpleNamespace(llm=SimpleNamespace(model="gpt-4"))
    )
    ai = Ai(computer)

    with mock.patch(
        "interpreter.core.computer.ai.ai.split_into_chunks", return_value=["c"]
    ):
        with mock.patch(
            "interpreter.core.computer.ai.ai.query_map_chunks", return_value=["r"]
        ) as mapper:
            with mock.patch(
                "interpreter.core.computer.ai.ai.query_reduce_chunks", return_value="final"
            ) as reducer:
                ai.summarize("a long body")

    # query_map_chunks(chunks, llm, query) / query_reduce_chunks(resp, llm, size, reduce_query)
    map_prompt = mapper.call_args.args[2]
    reduce_prompt = reducer.call_args.args[3]
    assert map_prompt != reduce_prompt
    assert "summarize it into a concise abstract" in map_prompt
    assert "merging them into one unified" in reduce_prompt


def test_chat_accumulates_every_content_chunk():
    """chat() concatenates the content of each streamed chunk.

    The provider streams a reply in pieces; dropping any of them returns a
    truncated answer that looks like a complete one.
    """
    llm = mock.Mock()
    llm.run.return_value = [
        {"content": "Hello, "},
        {"type": "active_line"},
        {"content": "world"},
    ]
    computer = SimpleNamespace(interpreter=SimpleNamespace(llm=llm))
    ai = Ai(computer)

    assert ai.chat("hi") == "Hello, world"


def test_chat_sends_the_system_message_before_the_user_text():
    """The first message is the assistant persona, then the user's text."""
    llm = mock.Mock()
    llm.run.return_value = [{"content": "ok"}]
    computer = SimpleNamespace(interpreter=SimpleNamespace(llm=llm))
    ai = Ai(computer)

    ai.chat("remember this")

    messages = llm.run.call_args.args[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful AI assistant."
    assert messages[1] == {"role": "user", "type": "message", "content": "remember this"}


def test_chat_appends_a_base64_image_message():
    """A base64 image is added as its own message after the text."""
    llm = mock.Mock()
    llm.run.return_value = [{"content": "an image"}]
    computer = SimpleNamespace(interpreter=SimpleNamespace(llm=llm))
    ai = Ai(computer)

    ai.chat("what is this?", base64="QUJD")

    messages = llm.run.call_args.args[0]
    assert messages[-1] == {
        "role": "user",
        "type": "image",
        "format": "base64",
        "content": "QUJD",
    }


def test_chat_omits_the_image_message_without_base64():
    """No image message is added when base64 is absent or empty.

    An empty string is falsy, so passing "" must not produce an empty image
    message that the provider would reject.
    """
    llm = mock.Mock()
    llm.run.return_value = [{"content": "ok"}]
    computer = SimpleNamespace(interpreter=SimpleNamespace(llm=llm))
    ai = Ai(computer)

    ai.chat("hi", base64="")

    assert len(llm.run.call_args.args[0]) == 2
