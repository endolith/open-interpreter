from interpreter.core.llm.utils.cache_aware_trim import (
    _count_message_tokens,
    cache_aware_trim,
)


SYSTEM = "You are a helpful assistant."


def _big_user(ts, marker):
    """A user message large enough to dominate the token budget."""
    return {"role": "user", "content": f"[{ts}] {marker} " + "x" * 4000}


def _placeholder(out):
    """Return the omission placeholder message, or None if there is none."""
    return next(
        (
            m
            for m in out
            if isinstance(m.get("content"), str) and "omitted" in m["content"]
        ),
        None,
    )


def test_no_trim_when_under_token_limit():
    """A prompt that fits inside the budget is returned untouched, without an
    omission placeholder, so short conversations are never altered."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = cache_aware_trim(messages, SYSTEM, token_limit=10000)
    assert out[0] == {"role": "system", "content": SYSTEM}
    assert out[1:] == messages
    assert _placeholder(out) is None


def test_trims_to_retention_target_dropping_whole_turns():
    """When the prompt outgrows the budget, a *variable* number of oldest whole
    turns is dropped so the retained tail sits at (or under) retention_ratio of
    the budget, the cut lands on a user boundary, and the placeholder notes how
    many messages were removed."""
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        {"role": "assistant", "content": "answer one"},
        _big_user("2026-08-11 09:00", "second"),
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "[2026-08-12 09:00] current prompt"},
    ]
    out = cache_aware_trim(messages, SYSTEM, token_limit=300)  # default ratio 0.8
    assert out[0] == {"role": "system", "content": SYSTEM}
    note = _placeholder(out)
    assert note is not None
    assert note["role"] == "user"
    assert note["content"] == (
        "[… 4 messages omitted from 2026-08-10 09:00 "
        "to 2026-08-11 09:00 to fit context window …]"
    )
    # Whole messages are kept — the retained tail is an exact suffix of the input.
    assert out[2:] == messages[4:]
    assert _count_message_tokens(out[2:], None) <= 300


def test_omission_note_count_grows_and_last_timestamp_updates():
    """Across successive truncations of a growing conversation the placeholder
    count increases and the trailing timestamp advances, while the leading
    timestamp stays fixed — so the model is told exactly how much older history
    is missing from the current request."""
    base = [
        _big_user("2026-08-10 09:00", "first"),
        {"role": "assistant", "content": "answer one"},
        _big_user("2026-08-11 09:00", "second"),
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "[2026-08-12 09:00] current prompt"},
    ]
    out1 = cache_aware_trim(base, SYSTEM, token_limit=300)
    note1 = _placeholder(out1)["content"]
    assert "4 messages omitted from 2026-08-10 09:00 to 2026-08-11 09:00" in note1

    grown = base + [
        _big_user("2026-08-13 09:00", "third"),
        {"role": "assistant", "content": "answer three"},
    ]
    out2 = cache_aware_trim(grown, SYSTEM, token_limit=300)
    note2 = _placeholder(out2)["content"]
    # Same leading timestamp, larger count, newer trailing timestamp.
    assert "5 messages omitted from 2026-08-10 09:00 to 2026-08-12 09:00" in note2


def test_cut_never_splits_a_tool_call_turn():
    """The cut always lands on a user message, so an assistant message holding a
    function/tool call and its result are dropped (or kept) together; a split
    would leave an orphaned tool result that strict providers reject with 400."""
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "execute", "arguments": '{"code":"print(1)"}'},
        },
        {"role": "function", "name": "execute", "content": "1"},
        {"role": "user", "content": "[2026-08-11 09:00] follow-up"},
        {"role": "assistant", "content": "done"},
    ]
    out = cache_aware_trim(messages, SYSTEM, token_limit=300)
    note = _placeholder(out)
    assert note is not None
    assert note["role"] == "user"
    retained = out[2:]
    # The retained history starts at a user boundary, never on an orphaned
    # function/tool result.
    assert retained[0]["role"] == "user"
    # The assistant tool-call and its function result were dropped together.
    assert not any("function_call" in m for m in retained)
    assert not any(m.get("role") == "function" for m in retained)
    assert retained == messages[3:]


def test_never_drops_the_current_prompt():
    """Even when the budget is far smaller than the history, the newest user
    message (the current prompt) is always kept — dropping it would make the
    model answer without seeing the actual question."""
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        {"role": "assistant", "content": "a" * 4000},
        {"role": "user", "content": "[2026-08-12 09:00] current prompt"},
    ]
    out = cache_aware_trim(messages, SYSTEM, token_limit=50)
    assert out[2]["content"] == "[2026-08-12 09:00] current prompt"
    assert len(out) == 3  # system + placeholder + the current prompt
    assert "2 messages omitted" in out[1]["content"]


def test_retention_ratio_controls_how_aggressively_history_is_dropped():
    """A smaller retention ratio trims further below the budget, dropping more
    old turns; a ratio of 1.0 only trims down to the limit itself."""
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        {"role": "assistant", "content": "answer one"},
        _big_user("2026-08-11 09:00", "second"),
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "[2026-08-12 09:00] medium " + "y" * 800},
        {"role": "assistant", "content": "answer three"},
        {"role": "user", "content": "[2026-08-13 09:00] current"},
    ]
    out_loose = cache_aware_trim(
        messages, SYSTEM, token_limit=500, retention_ratio=1.0
    )
    out_aggressive = cache_aware_trim(
        messages, SYSTEM, token_limit=500, retention_ratio=0.3
    )
    assert len(out_loose) > len(out_aggressive)
    assert "4 messages omitted" in _placeholder(out_loose)["content"]
    assert "6 messages omitted" in _placeholder(out_aggressive)["content"]


def test_count_message_tokens_counts_reasoning_content():
    """Thinking-model turns store huge reasoning blobs in `reasoning_content`;
    earlier versions only counted `content`, so a ~10k-char reasoning blob was
    counted as a handful of tokens and cache-aware trimming never fired.  The
    counter must include it so these turns are trimmed correctly."""
    plain = [{"role": "assistant", "content": "ok"}]
    thinking = [
        {"role": "assistant", "content": "ok", "reasoning_content": "r" * 10000}
    ]
    plain_tokens = _count_message_tokens(plain, None)
    thinking_tokens = _count_message_tokens(thinking, None)
    assert thinking_tokens > plain_tokens * 10


def test_leaves_history_untouched_without_a_user_boundary():
    """If the history has no user message at all there is no safe place to cut
    (any cut could orphan a tool result), so it is returned unchanged without a
    placeholder."""
    messages = [
        {"role": "assistant", "content": "x" * 4000},
        {"role": "function", "name": "execute", "content": "y" * 4000},
    ]
    out = cache_aware_trim(messages, SYSTEM, token_limit=10)
    assert out == [{"role": "system", "content": SYSTEM}] + messages
    assert _placeholder(out) is None


def test_placeholder_uses_singular_for_one_message():
    """The placeholder message reads naturally for a single dropped message."""
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        {"role": "user", "content": "[2026-08-11 09:00] current"},
    ]
    out = cache_aware_trim(messages, SYSTEM, token_limit=50)
    assert "1 message omitted" in _placeholder(out)["content"]
