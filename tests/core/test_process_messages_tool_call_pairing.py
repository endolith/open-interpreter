"""Tests for tool_call/tool-response pairing in process_messages.

Every OpenAI-compatible endpoint requires a strict invariant: each tool_call
must be answered by exactly one tool message carrying its id, and ids must be
unique. A request that breaks either is rejected outright. On the OpenCode Go
relay the rejection is a bare "400" with no body, and the identical retry then
succeeds -- because by then the offending entry has been consumed, so the fault
looked intermittent when it was deterministic.
"""

from interpreter.core.llm.run_tool_calling_llm import process_messages


def _tool_call(call_id, code="print(1)"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "execute", "arguments": f'{{"language":"python","code":"{code}"}}'},
    }


def _assert_well_formed(processed):
    """Every tool_call has exactly one response, ids unique and in order."""
    calls = [call for m in processed for call in (m.get("tool_calls") or [])]
    responses = [m for m in processed if m.get("role") == "tool"]
    call_ids = [call["id"] for call in calls]
    assert len(call_ids) == len(set(call_ids)), f"duplicate tool_call ids: {call_ids}"
    assert len(calls) == len(responses), (
        f"{len(calls)} tool_calls but {len(responses)} tool responses; every call must be answered exactly once"
    )
    assert {m["tool_call_id"] for m in responses} == set(call_ids), (
        "each response must reference a call that exists in the request"
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


def test_truly_orphaned_function_result_still_synthesizes_a_call():
    """With no originating call anywhere, a call is synthesized so the request stays valid.

    A role="function" result whose call is genuinely absent (a stored error
    response for a call we never kept) must still be preceded by an assistant
    with tool_calls, or providers reject the tool message outright.
    """
    processed = process_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "function", "name": "execute", "content": "out"},
        ],
        model="openai/deepseek-v4.1-flash",
    )

    _assert_well_formed(processed)
    assert any(m.get("tool_calls") for m in processed), "an orphan still needs a preceding call to attach its result to"


def test_synthesized_id_does_not_collide_with_an_existing_call():
    """A generated id must not reuse one already present in the request.

    last_tool_id counts only the calls OI converted, so it restarts at 1 and
    would hand a genuine "toolu_1" to the synthetic call. Duplicate ids make the
    gateway reject the entire request.
    """
    processed = process_messages(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("toolu_1")],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "content": "answered"},
            # Orphan follows a completed call, so it must get a fresh id.
            {"role": "function", "name": "execute", "content": "later"},
        ],
        model="openai/deepseek-v4.1-flash",
    )

    _assert_well_formed(processed)


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
