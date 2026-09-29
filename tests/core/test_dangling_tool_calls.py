"""Regression tests for tool calls that never received a response.

An interrupted turn leaves the conversation history holding an assistant message
with ``tool_calls`` and no matching ``tool`` response. Replaying that history --
which is what the first turn after a ``--conversations`` resume does -- produces
a request that every OpenAI-compatible validator rejects as invalid (a 400),
because each ``tool_calls`` entry must be followed by a ``tool`` message carrying
the same ``tool_call_id``.

The sibling direction, a tool response with no call, is already covered in
``test_process_messages_tool_call_pairing.py``.
"""

import pytest

from interpreter.core.llm.run_tool_calling_llm import process_messages

MODEL = "openai/deepseek-v4.1-flash"


def _call(call_id, arguments="{}"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "execute", "arguments": arguments},
    }


def _audit(messages):
    """Return (call_ids, response_ids) so pairing can be asserted directly."""
    calls = [c["id"] for m in messages for c in (m.get("tool_calls") or [])]
    responses = [m.get("tool_call_id") for m in messages if m.get("role") == "tool"]
    return calls, responses


def _run(history):
    return process_messages([dict(m) for m in history], model=MODEL)


def test_dangling_tool_call_is_given_a_response():
    """A call with no response is the state an interrupted turn leaves behind.

    Without repair the request is structurally invalid, so the provider rejects
    it before the model ever sees the conversation.
    """
    out = _run(
        [
            {"role": "user", "content": "read the configs"},
            {"role": "assistant", "content": "", "tool_calls": [_call("call_1")]},
        ]
    )
    calls, responses = _audit(out)
    assert calls == responses == ["call_1"]


def test_synthetic_response_says_the_call_never_ran():
    """The inserted response must not read as a result.

    A response implying the call succeeded would let the model reason from an
    outcome that never happened, which is worse than the 400 it replaces.
    """
    out = _run(
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [_call("call_1")]},
        ]
    )
    synthetic = [m for m in out if m.get("role") == "tool"][-1]
    assert "never executed" in synthetic["content"]
    assert "Interrupted" in synthetic["content"]


def test_response_keeps_the_original_tool_call_id():
    """The pairing is only valid if the id matches the call exactly."""
    out = _run(
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [_call("call_abc123")]},
        ]
    )
    assert [m["tool_call_id"] for m in out if m.get("role") == "tool"] == ["call_abc123"]


def test_dangling_call_at_end_of_long_history_is_repaired():
    """The resume shape: the unfinished turn is last, after several good turns."""
    out = _run(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "", "tool_calls": [_call("z9")]},
        ]
    )
    calls, responses = _audit(out)
    assert calls == responses == ["z9"]


def test_only_the_unanswered_call_is_synthesised():
    """With two calls and one real response, exactly one response is added.

    Adding a second would send a duplicate result for a call that already
    answered, which strict validators also reject.
    """
    out = _run(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("c1"), _call("c2")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "real result"},
        ]
    )
    calls, responses = _audit(out)
    assert sorted(calls) == sorted(responses) == ["c1", "c2"]
    assert len(responses) == 2, "a duplicate response for c1 would be invalid"


def test_real_response_precedes_synthetic_one():
    """Ordering follows the call order, with synthetic responses appended last.

    Putting a synthetic result in front of a real one would misrepresent which
    call produced what.
    """
    out = _run(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("c1"), _call("c2")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "real result"},
        ]
    )
    ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
    assert ids == ["c1", "c2"]


def test_already_paired_call_is_left_untouched():
    """The ordinary path must be unchanged: no extra message, same content."""
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "1"},
    ]
    out = _run(history)
    assert len(out) == len(history)
    assert out[-1]["content"] == "1"


def test_conversation_without_tool_calls_is_unchanged():
    """A plain text turn must not gain a synthetic response."""
    out = _run(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert not any(m.get("role") == "tool" for m in out)


def test_dangling_call_with_no_id_is_not_invented_onto():
    """A call with no id cannot be answered, since there is nothing to match on.

    Firing a response with a made-up id would be worse than leaving it, so the
    entry is skipped rather than papered over.
    """
    call = _call("c1")
    del call["id"]
    out = _run(
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
        ]
    )
    assert not any(m.get("role") == "tool" for m in out)


@pytest.mark.parametrize(
    "ids",
    [
        ["a"],
        ["a", "b"],
        ["a", "b", "c"],
    ],
)
def test_any_number_of_dangling_calls_is_fully_answered(ids):
    """Whatever the model emitted, every answered id is accounted for.

    Parametrised because the count is the thing that varies; a single case would
    not have caught the duplicate-response bug above.
    """
    out = _run(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call(i) for i in ids],
            },
        ]
    )
    calls, responses = _audit(out)
    assert calls == responses == ids
