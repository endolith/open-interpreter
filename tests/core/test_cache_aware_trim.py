from interpreter.core.llm.utils.cache_aware_trim import (
    _count_message_tokens,
    _count_text,
    _extract_ts,
    _find_safe_cut,
    _get_encoding,
    _image_token_estimate,
    _omission_note,
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
    the budget, the cut lands on a safe boundary, and the placeholder notes how
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
        "[… 3 messages omitted from 2026-08-10 09:00 "
        "to 2026-08-11 09:00 to fit context window …]"
    )
    # Whole messages are kept — the retained tail is an exact suffix of the input.
    # The cut lands on the plain-assistant "answer two" (a safe head), so the
    # model keeps the last assistant reply as well as the current prompt.
    assert out[2:] == messages[3:]
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
    assert "3 messages omitted from 2026-08-10 09:00 to 2026-08-11 09:00" in note1

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
    assert "3 messages omitted" in _placeholder(out_loose)["content"]
    assert "5 messages omitted" in _placeholder(out_aggressive)["content"]


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


def test_image_tokens_charged_estimate_not_base64_size():
    """A base64 image must not be counted by its payload length, or a single
    screenshot (megabytes of characters when base64-encoded) would look larger
    than the whole context window and trigger a trim that wipes the entire
    conversation.  Image parts are charged a model-aware per-image token
    estimate instead, so the trim ignores the payload size."""
    tiny = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + "A" * 100,
                    "detail": "low",
                },
            }
        ],
    }
    huge = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + "A" * 2000000,
                    "detail": "low",
                },
            }
        ],
    }
    assert _count_message_tokens([tiny], None) == _count_message_tokens([huge], None)
    # The image cost is a bounded estimate plus per-message framing, not
    # proportional to the payload.  (2M base64 chars would otherwise count as
    # hundreds of thousands of "tokens".)  With the model unknown the counter
    # assumes the conservative low-detail floor (2,833), not the 85 minimum.
    assert _count_message_tokens([huge], None) < 3000


def test_image_in_history_does_not_wreck_the_conversation():
    """Adding an image (e.g. via the view_image tool) mid-conversation must not
    make the trim drop everything else: before the fix the base64 data URL was
    counted as text tokens, so a single image overflowed the budget and the
    whole conversation history was trimmed away."""
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        {"role": "assistant", "content": "answer one"},
        _big_user("2026-08-11 09:00", "second"),
        {"role": "assistant", "content": "answer two"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + "A" * 2000000,
                        "detail": "low",
                    },
                }
            ],
        },
        {"role": "assistant", "content": "I can see the image"},
    ]
    # token_limit far larger than the conversation + image at fixed cost, so
    # nothing should be trimmed even with the giant payload in history.
    out = cache_aware_trim(messages, SYSTEM, token_limit=100000)
    assert _placeholder(out) is None
    assert out[1:] == messages


def test_cut_lands_on_plain_assistant_boundary_to_keep_full_budget():
    """A long tail of assistant-reasoning / function / tool messages contains no
    user boundary, so cutting only on user messages would keep dropping past
    the point where the tail first fits — discarding nearly the whole
    conversation and wasting the retention budget.  The cut is allowed on a
    plain assistant message (no dangling tool call), keeping as much of the
    budget as possible while the current prompt is still retained."""
    messages = [
        {"role": "user", "content": "[2026-08-01 09:00] start"},
    ]
    for _ in range(12):
        messages.append(
            {"role": "assistant", "content": "ok", "reasoning_content": "r" * 4000}
        )
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "function_call": {"name": "execute", "arguments": "{}"},
            }
        )
        messages.append(
            {"role": "function", "name": "execute", "content": "output " + "x" * 3000}
        )
    messages.append({"role": "user", "content": "[2026-08-02 09:00] current prompt"})

    out = cache_aware_trim(messages, SYSTEM, token_limit=10000, retention_ratio=0.8)
    note = _placeholder(out)
    assert note is not None
    retained = out[2:]
    # The retained tail actually uses the budget instead of a sliver of it.
    assert _count_message_tokens(retained, None) > 4000  # > half of the 8000 target
    assert retained[-1]["role"] == "user"  # current prompt never dropped
    # The head is a safe boundary — never an orphaned function/tool result.
    assert retained[0]["role"] in ("user", "assistant")
    assert not retained[0].get("function_call")
    assert not retained[0].get("tool_calls")


def test_cut_never_orphans_an_assistant_tool_call():
    """Cutting right before an assistant message that holds tool_calls would
    orphan the tool call from its result (strict providers 400 on that shape),
    so the cut must skip past tool-call heads even when they are assistants."""
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "view_image", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "image added"},
        {"role": "user", "content": "[2026-08-11 09:00] current prompt"},
    ]
    out = cache_aware_trim(messages, SYSTEM, token_limit=300)
    note = _placeholder(out)
    assert note is not None
    retained = out[2:]
    assert retained[0]["role"] == "user"
    assert not any(m.get("tool_calls") for m in retained)
    assert not any(m.get("role") == "tool" for m in retained)
    assert retained == messages[3:]


def test_count_text_ignores_non_text_values():
    """Non-text scalars cost zero tokens instead of breaking the count.

    Message dicts can carry numbers, booleans, or nulls (for example a
    `tool_call` index), so the counter must skip them rather than raise and
    abort the whole trim.
    """
    encoding = _get_encoding(None)
    assert _count_text(42, encoding) == 0
    assert _count_text(None, encoding) == 0
    assert _count_text([42, None], encoding) == 0


def test_find_safe_cut_counts_when_costs_are_not_precomputed():
    """The cut helper stays usable without precomputed per-message costs.

    Callers with a conversion cache pass costs in, but the default path must
    count for itself and land on the same boundary.
    """
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        {"role": "assistant", "content": "a"},
        _big_user("2026-08-11 09:00", "second"),
    ]
    costs = [_count_message_tokens([m], None) for m in messages]
    assert _find_safe_cut(messages, 1000, None) == _find_safe_cut(
        messages, 1000, None, costs=costs
    )


def test_extract_ts_reads_text_part_in_multipart_message():
    """Multipart user messages still yield their `[YYYY-MM-DD HH:MM]` prefix.

    Once vision is used, user content becomes a list of text/image parts, and
    only the text part carries the timestamp the omission placeholder quotes.
    """
    assert (
        _extract_ts(
            {
                "role": "user",
                "content": [{"type": "text", "text": "[2026-11-12 07:00] hi"}],
            }
        )
        == "2026-11-12 07:00"
    )
    # Content that is neither a string nor a list of parts carries no prefix.
    assert _extract_ts({"role": "user", "content": None}) is None
    # No text part at all, and a text part without a timestamp, both yield None.
    assert (
        _extract_ts(
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:..."}}],
            }
        )
        is None
    )
    assert (
        _extract_ts({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        is None
    )


def test_retention_ratio_none_applies_default():
    """Omitting the retention ratio selects the documented 0.8 default."""
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        _big_user("2026-08-11 09:00", "second"),
        _big_user("2026-08-12 09:00", "third"),
    ]
    assert cache_aware_trim(
        messages, SYSTEM, token_limit=500, retention_ratio=None
    ) == cache_aware_trim(messages, SYSTEM, token_limit=500)


def test_precomputed_message_costs_match_the_counted_path():
    """Supplied per-message costs are used verbatim, skipping a second pass."""
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        _big_user("2026-08-11 09:00", "second"),
        {"role": "assistant", "content": "a"},
        _big_user("2026-08-12 09:00", "third"),
    ]
    costs = [_count_message_tokens([m], None) for m in messages]
    assert cache_aware_trim(
        messages, SYSTEM, token_limit=500, message_costs=costs
    ) == cache_aware_trim(messages, SYSTEM, token_limit=500)


def test_count_text_counts_special_token_literals_as_text():
    """Literal special-token text must not raise while counting.

    A user message can contain `<|endoftext|>`, which `encoding.encode` treats
    as markup and rejects with ValueError.  The trim caller would then send
    the full untrimmed history, so the counter must strip special-token
    semantics and charge those characters as ordinary text.
    """
    encoding = _get_encoding(None)
    assert _count_text("<|endoftext|>", encoding) > 0
    assert _count_text("<|endoftext|>", encoding) == len(
        encoding.encode("<|endoftext|>", disallowed_special=())
    )


def test_image_token_estimate_is_model_aware():
    """Image cost estimates follow the model instead of a single flat rate.

    `gpt-4o-mini` bills a low-detail image at a much higher base cost than the
    85-token floor, so charging every low-detail image 85 would let two images
    look like they fit a small window and still overflow at the provider.
    Unknown or unrecognised model names (including ones that merely contain a
    known family name) must also use the conservative bound, never the cheap one.
    """
    assert _image_token_estimate("low", None) == 2833
    assert _image_token_estimate("low", "gpt-4o-mini") == 2833
    assert _image_token_estimate("low", "openai/gpt-4o-mini") == 2833
    assert _image_token_estimate("low", "gpt-4o-mini-2024-07-18") == 2833
    assert _image_token_estimate("low", "gpt-4o") == 85
    assert _image_token_estimate("low", "openai/gpt-4o") == 85
    assert _image_token_estimate("low", "gpt-4o-2024-05-13") == 85
    assert _image_token_estimate("low", "openai/gpt-4o-2024-05-13") == 85
    assert _image_token_estimate("low", "claude-3-5-sonnet-latest") == 2833
    assert _image_token_estimate("low", "some-unknown-model") == 2833
    assert _image_token_estimate("low", "some-unknown-gpt-4o-compatible") == 2833
    assert _image_token_estimate("low", "gpt-4o-compatible") == 2833
    assert _image_token_estimate("high", "gpt-4o") == 1105
    assert _image_token_estimate(None, "gpt-4o") == 1105


def test_trim_budget_includes_system_prompt():
    """The system prompt counts against the hard limit, not just the messages.

    `_find_safe_cut` compares only message costs with its target.  A system
    prompt that costs ~100 tokens can make a conversation that fits on paper
    (two ~100-token user messages with a 250-token limit) overflow the real
    request unless the trim subtracts it before choosing the cut.
    """
    system = "word " * 90  # ~100 tokens
    messages = [
        {"role": "user", "content": "word " * 90},
        {"role": "user", "content": "word " * 90},
    ]
    full = [{"role": "system", "content": system}] + messages
    assert _count_message_tokens(full, None) > 250

    out = cache_aware_trim(messages, system, token_limit=250, retention_ratio=1.0)
    assert _placeholder(out) is not None
    assert _count_message_tokens(out, None) <= 250


def test_placeholder_overflow_advances_the_cut_until_the_request_fits():
    """The omission note is part of the request and can push a fitting cut over.

    A cut sized for the retention target can still exceed the hard limit once
    the placeholder describing the omitted messages is inserted.  The trim
    must advance to the next safe boundary and recheck the complete result,
    while never dropping the newest user message.
    """
    system = "s"
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        {"role": "assistant", "content": "answer one"},
        _big_user("2026-08-11 09:00", "second"),
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "[2026-08-12 09:00] current prompt"},
    ]
    last_user = len(messages) - 1
    minimum = _count_message_tokens(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": _omission_note(messages, last_user)},
        ]
        + messages[last_user:],
        None,
    )
    for limit in (2300, 2400, 2600, 3200):
        if limit < minimum:
            continue
        out = cache_aware_trim(messages, system, token_limit=limit, retention_ratio=1.0)
        assert _count_message_tokens(out, None) <= limit
        assert out[-1] == messages[last_user]


def test_reserve_tokens_leaves_room_for_the_runner_additions():
    """Reserved tokens shrink the effective budget before the cut is chosen.

    The tool-calling runner attaches its schema after trimming, so the trim
    must leave room for it: a conversation that fits the raw limit can need
    trimming once the runner's overhead is reserved.
    """
    messages = [
        _big_user("2026-08-10 09:00", "first"),
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "[2026-08-11 09:00] current prompt"},
    ]
    # Size the limit from the real conversation so the unreserved budget
    # genuinely fits; the reserved one then has to trim the same history.
    full = _count_message_tokens([{"role": "system", "content": SYSTEM}] + messages, None)
    limit = full + 10
    assert _placeholder(cache_aware_trim(messages, SYSTEM, token_limit=limit)) is None

    reserved = cache_aware_trim(
        messages, SYSTEM, token_limit=limit, reserve_tokens=200
    )
    assert _placeholder(reserved) is not None
    assert _count_message_tokens(reserved, None) <= limit - 200
