"""Tests for tool_call/tool-response pairing in process_messages.

Every OpenAI-compatible endpoint requires a strict invariant, scoped to each
turn: every tool_call must be answered by exactly one tool message carrying its
id, and ids must be unique *within that turn*. Models routinely reuse the same
id on every turn (typically "toolu_1"), so id reuse across turns is normal and
must not be treated as a collision.

A request that breaks the per-turn invariant is rejected outright. On the
OpenCode Go relay the rejection is a bare "400", and because the offending entry
was consumed by the time the user retried, the fault looked intermittent while
being deterministic.
"""

import json

from interpreter.core.llm.run_tool_calling_llm import process_messages


def _tool_call(call_id, code="print(1)"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "execute", "arguments": f'{{"language":"python","code":"{code}"}}'},
    }


def _turns(processed):
    """Split a processed message list into turns, each starting at a user message."""
    turns, current = [], []
    for message in list(processed) + [{"role": "user", "content": "__end__"}]:
        if message.get("role") == "user":
            if current:
                turns.append(current)
            current = []
        else:
            current.append(message)
    return turns


def _assert_well_formed(processed):
    """Per turn: one response per call, and unique ids within the turn.

    Uniqueness is checked per turn, not globally. The same id on two different
    turns is normal (models reuse "toolu_1" every turn); a duplicate inside one
    turn is what providers reject.
    """
    for index, turn in enumerate(_turns(processed)):
        calls = [call for m in turn for call in (m.get("tool_calls") or [])]
        responses = [m for m in turn if m.get("role") == "tool"]
        call_ids = [call["id"] for call in calls]
        assert len(call_ids) == len(set(call_ids)), f"turn {index}: duplicate tool_call ids {call_ids}"
        assert len(calls) == len(responses), (
            f"turn {index}: {len(calls)} tool_calls but {len(responses)} tool "
            "responses; every call must be answered exactly once"
        )
        assert {m["tool_call_id"] for m in responses} == set(call_ids), (
            f"turn {index}: each response must reference a call in the same turn"
        )


def test_out_of_order_function_result_pairs_with_real_call():
    """A function result whose call is already in the request must not fabricate another.

    The stored history can hold a result as role="function" while the assistant
    entry that made the call carries tool_calls. Treating that as orphaned
    appended a second, synthetic tool_call with a freshly generated id, so the
    request carried two calls (one duplicated id, one invented) and a single
    response -- which the gateway rejects. The result belongs to the call that
    is already there.
    """
    processed = process_messages(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "run it",
                "tool_calls": [_tool_call("toolu_1")],
            },
            {"role": "function", "name": "execute", "content": "1"},
        ],
        model="openai/deepseek-v4.1-flash",
    )

    _assert_well_formed(processed)
    assistant = [m for m in processed if m.get("tool_calls")]
    assert len(assistant) == 1
    assert len(assistant[0]["tool_calls"]) == 1, "no second synthetic call may be invented for an already-present one"
    assert processed[-1]["tool_call_id"] == "toolu_1", "the result must reference the real call id"
    assert "Automated tool call" not in processed[-1].get("content", "")


def test_id_reused_across_turns_does_not_trigger_a_synthetic_call():
    """A result is paired with the CURRENT turn's call even when the id repeats.

    Models reuse the same tool_call id on every turn (typically "toolu_1"), so
    the previous turn's response for that id is already in the request when the
    new result arrives. Matching on "has this id been answered anywhere?" made
    the current call look already-answered, so a spurious synthetic call was
    appended and the real call left unanswered -- the exact shape the gateway
    rejects, and it recurred on every turn after the first tool call.
    """
    processed = process_messages(
        [
            {"role": "user", "content": "py hello world"},
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("toolu_1")]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "hello"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "again in cmd"},
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("toolu_1", "echo hi")]},
            {"role": "function", "name": "execute", "content": "cmd output"},
        ],
        model="openai/deepseek-v4.1-flash",
    )

    _assert_well_formed(processed)
    for message in processed:
        for call in message.get("tool_calls") or []:
            assert "Automated tool call" not in call["function"]["arguments"], (
                "the current turn's call must be paired, not duplicated"
            )


def test_consecutive_turns_all_stay_paired():
    """Every turn of a multi-turn tool session remains individually well formed."""
    messages = []
    for i, (code, output) in enumerate([("print(1)", "hello"), ("echo hi", "cmd out"), ("Write-Output x", "ps out")]):
        messages += [
            {"role": "user", "content": f"turn {i}"},
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("toolu_1", code)]},
            {"role": "function", "name": "execute", "content": output},
            {"role": "assistant", "content": "ok"},
        ]

    processed = process_messages(messages, model="openai/deepseek-v4.1-flash")

    assert len(_turns(processed)) == 3
    _assert_well_formed(processed)
    synthesized = [
        call
        for message in processed
        for call in (message.get("tool_calls") or [])
        if "Automated tool call" in call["function"]["arguments"]
    ]
    assert not synthesized, "no turn may need a synthetic call"


def test_process_messages_does_not_mutate_its_input():
    """Caller-owned message dicts must come back untouched.

    process_messages() rewrites role and tool_call_id in place, and the same
    dicts are reused for the next request. Mutating them writes this turn's
    pairing into stored history, so a later turn sends a tool_call_id that
    belongs to an earlier turn and the request is invalid from then on.
    """
    messages = [
        {"role": "user", "content": "again in cmd"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("toolu_1")]},
        {"role": "function", "name": "execute", "content": "cmd output"},
    ]
    snapshot = json.loads(json.dumps(messages, sort_keys=True))

    process_messages(messages, model="openai/deepseek-v4.1-flash")

    assert json.loads(json.dumps(messages, sort_keys=True)) == snapshot, (
        "process_messages must not modify the messages it is given"
    )


def test_truly_orphaned_function_result_becomes_a_message():
    """An orphaned result is surfaced as a message, never a fabricated execute call.

    A synthetic execute(pass) taught the model to emit python calls with no
    code. With no stored call there is nothing for the provider to pair, so the
    output becomes a message and the request stays valid.
    """
    processed = process_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "function", "name": "execute", "content": "out"},
        ],
        model="openai/deepseek-v4.1-flash",
    )

    _assert_well_formed(processed)
    assert not any(m.get("tool_calls") for m in processed)
    assert any("out" in str(m.get("content", "")) for m in processed)


def test_orphaned_tool_result_becomes_a_message():
    """A modern tool result with no stored call must not get a fake execute(pass)."""
    processed = process_messages(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "tool",
                "tool_call_id": "call_x",
                "content": "Invalid execute call: code is empty",
            },
        ],
        model="opencode_go/deepseek-v4.1-flash",
    )

    _assert_well_formed(processed)
    assert not any(m.get("tool_calls") for m in processed)
    assert any(
        "Invalid execute call" in str(m.get("content", "")) for m in processed
    )


def test_orphan_after_a_completed_call_becomes_a_message():
    """An orphan following a finished call must not add a second fabricated call."""
    processed = process_messages(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("toolu_1")],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "content": "answered"},
            {"role": "function", "name": "execute", "content": "later"},
        ],
        model="openai/deepseek-v4.1-flash",
    )

    _assert_well_formed(processed)
    calls = [call for m in processed for call in (m.get("tool_calls") or [])]
    assert len(calls) == 1, "only the real call should remain"
    assert any("later" in str(m.get("content", "")) for m in processed)


def test_multiple_out_of_order_results_pair_one_to_one():
    """Each result in a reordered run maps to a distinct call, not all to the first."""
    processed = process_messages(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("a1"), _tool_call("a2")],
            },
            {"role": "function", "content": "second"},
            {"role": "function", "content": "first"},
        ],
        model="openai/deepseek-v4.1-flash",
    )

    _assert_well_formed(processed)
    responses = [m["tool_call_id"] for m in processed if m.get("role") == "tool"]
    assert sorted(responses) == ["a1", "a2"], "two results must answer two distinct calls, not the same one twice"


def test_ordered_call_and_result_are_left_alone():
    """The normal, already-valid shape is unchanged by the pairing logic."""
    processed = process_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("toolu_1")]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "out"},
        ],
        model="openai/deepseek-v4.1-flash",
    )

    _assert_well_formed(processed)
    assert len(processed) == 3, "no synthetic assistant may be inserted"


def test_mistral_id_format_is_preserved_for_paired_results():
    """Pairing must not rewrite an id the model supplied into another format.

    Mistral requires tool call ids to match ^[a-zA-Z0-9]{9}$ while other
    providers use the toolu_N form. A stored id is passed through as-is, so a
    result answering a real Mistral call keeps that call's id.
    """
    mistral_call = {
        "id": "tool00001",
        "type": "function",
        "function": {"name": "execute", "arguments": "{}"},
    }
    processed = process_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [mistral_call]},
            {"role": "function", "name": "execute", "content": "out"},
        ],
        model="mistral/mistral-large",
    )

    _assert_well_formed(processed)
    assert processed[-1]["tool_call_id"] == "tool00001", "the model's own id must be preserved, not regenerated"
