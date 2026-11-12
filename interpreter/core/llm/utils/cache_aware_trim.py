import re

import tiktoken


_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]")

# Per-image token *estimates* used when the payload itself is not tokenized.
# They are not provider-contract numbers: OpenAI's low-detail base cost varies
# by model, and other providers derive image tokens from dimensions.  The
# conservative floors below avoid underestimating an image when the provider's
# exact accounting is unknown.
_IMAGE_TOKENS_LOW_DETAIL = 85
_IMAGE_TOKENS_LOW_DETAIL_CONSERVATIVE = 2833
_IMAGE_TOKENS_DEFAULT = 1105


def _get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _image_token_estimate(detail, model):
    """Estimated tokens a single image part adds to the request.

    Low-detail images are not uniformly cheap: OpenAI's documented base cost
    for `gpt-4o-mini` is much higher than the 85-token floor other models use.
    When the model (and therefore the provider) is unknown, assume the
    conservative floor so the trim can never underestimate image cost.
    """
    if detail != "low":
        return _IMAGE_TOKENS_DEFAULT
    if model is None or "gpt-4o-mini" in model.lower():
        return _IMAGE_TOKENS_LOW_DETAIL_CONSERVATIVE
    return _IMAGE_TOKENS_LOW_DETAIL


def _count_text(obj, encoding, model=None):
    """Recursively count the tokens in every string value of an OpenAI message dict.

    Walks nested dicts/lists so that `content` (string or multi-part),
    `reasoning_content`, `name`, `function_call` and `tool_calls` are all
    counted.  Earlier versions only counted `content`, which massively
    undercounted thinking-model turns: a 20k-character reasoning blob was
    counted as a handful of tokens, so cache-aware trimming never fired.

    Strings are counted as ordinary text (`disallowed_special=()`), not as
    special-token markup: a user writing `<|endoftext|>` would otherwise make
    `encoding.encode` raise, and the trim caller would fall back to sending the
    untrimmed history.

    Image parts are NOT counted by their base64 payload: a data URL can be
    megabytes of characters that the tokenizer would inflate into hundreds of
    thousands of "tokens", making a single screenshot look larger than the
    whole context window and wiping every other message from the request.  Each
    image_url part is instead charged a model-aware per-image token *estimate*
    (see ``_image_token_estimate``), which is what the API approximately bills.
    """
    if isinstance(obj, str):
        return len(encoding.encode(obj, disallowed_special=()))
    if isinstance(obj, dict):
        if obj.get("type") == "image_url":
            detail = (obj.get("image_url") or {}).get("detail")
            return _image_token_estimate(detail, model)
        return sum(_count_text(value, encoding, model) for value in obj.values())
    if isinstance(obj, list):
        return sum(_count_text(item, encoding, model) for item in obj)
    return 0


def _count_message_tokens(messages, model):
    """Approximate token count for a list of OpenAI-format messages.

    Uses the same per-message overhead as OpenAI's cookbook (3 tokens per
    message for role/structure framing, plus 3 tokens for reply priming at
    the end of the list).  Image parts are charged a model-aware per-image
    estimate (see ``_count_text``), not their base64 payload size, so images
    count as a bounded amount rather than a huge text blob.
    """
    encoding = _get_encoding(model)
    total = 3  # reply priming tokens added by the API
    for msg in messages:
        total += 3  # per-message role/structure overhead
        total += _count_text(
            {k: v for k, v in msg.items() if k != "role"}, encoding, model
        )
    return total


def _is_safe_cut(message):
    """A message that can head the retained history without breaking the API.

    The head may be a user message, or an assistant message that carries no
    dangling tool call — a plain text/reasoning assistant head is valid for
    chat APIs.  Cutting on a `function`/`tool` result or an assistant message
    holding `function_call`/`tool_calls` would orphan the tool call from its
    result, which strict providers reject with a 400.
    """
    role = message.get("role")
    if role == "user":
        return True
    if role == "assistant" and not message.get("function_call") and not message.get(
        "tool_calls"
    ):
        return True
    return False


def _last_user_index(messages):
    """Index of the newest user message, or -1 when the history has none.

    The newest user message is the current prompt: trimming must never drop it,
    even when it alone exceeds the budget.
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def _next_safe_cut(messages, cut, last_user):
    """Next safe boundary after `cut`, never past the current prompt.

    Returns None when there is no later boundary to advance to, which means the
    retained tail cannot be made smaller without dropping the current turn.
    """
    for i in range(cut + 1, last_user + 1):
        if _is_safe_cut(messages[i]):
            return i
    return None


def _find_safe_cut(messages, target_tokens, model, costs=None):
    """Index of the oldest message to *keep* after dropping the prefix.

    Returns `cut` such that dropping `messages[:cut]` brings the retained tail
    at or below `target_tokens`.  The cut always lands on a boundary that does
    not orphan a tool result — a `user` message, or a plain `assistant`
    message with no `function_call`/`tool_calls` — so a tool-call turn (an
    assistant message holding `function_call`/`tool_calls`, immediately
    followed by its `function`/`tool` results) is never split in two, and no
    orphaned tool result is ever left as the first retained message (strict
    providers reject both shapes with a 400).  The newest user message (the
    current prompt) is never dropped, even if it alone exceeds the target.

    Restricting the cut to `user` messages alone is too strict: a long run of
    assistant-reasoning / function / tool messages (a normal agentic
    debugging session) contains no user boundary, so once the tail first fits
    the budget the search keeps dropping messages all the way to the next user
    message — retaining a tiny sliver of the available budget and discarding
    nearly the whole conversation.  Permitting plain-assistant boundaries lets
    the trim keep as much of the budget as it can.

    `costs` optionally supplies precomputed per-message token counts (one per
    message, same order) so callers that already counted — e.g. an incremental
    conversion cache — are not charged a second full pass.

    `target_tokens` is a budget for the *messages* only; callers must subtract
    the system prompt and any placeholder they add from the hard limit before
    passing it here (``cache_aware_trim`` does, and then rechecks the complete
    result because the placeholder can push a fitting cut back over the limit).
    """
    if costs is None:
        costs = [_count_message_tokens([m], model) for m in messages]
    retained = sum(costs)

    last_user = _last_user_index(messages)
    if last_user < 0:
        # No user boundary anywhere: leave the history untouched rather than
        # risk orphaning a tool result.
        return 0

    for i, cost in enumerate(costs):
        if (
            retained <= target_tokens
            and _is_safe_cut(messages[i])
            and i <= last_user
        ):
            return i
        retained -= cost
    # Even dropping everything up to the current turn can't reach the target:
    # keep the current turn and nothing else.
    return last_user


def _extract_ts(message):
    """Pull the `[YYYY-MM-DD HH:MM]` timestamp prefix off a user message."""
    content = message.get("content")
    if isinstance(content, str):
        match = _TS_RE.match(content)
        if match:
            return match.group(1)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                match = _TS_RE.match(part.get("text", ""))
                if match:
                    return match.group(1)
    return None


def _omission_note(messages, cut):
    """Build the placeholder telling the model older messages were removed.

    `count` is the total number of messages omitted from the request; it grows
    (and the trailing timestamp advances) as later truncations drop more of the
    conversation, while the leading timestamp stays fixed at the oldest dropped
    message.
    """
    count = cut
    first_ts = None
    last_ts = None
    for message in messages[:cut]:
        ts = _extract_ts(message)
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            last_ts = ts
    noun = "message" if count == 1 else "messages"
    if first_ts and last_ts:
        return (
            f"[… {count} {noun} omitted from {first_ts} to {last_ts} "
            f"to fit context window …]"
        )
    return f"[… {count} {noun} omitted to fit context window …]"


def cache_aware_trim(
    messages,
    system_message,
    token_limit,
    retention_ratio=0.8,
    model=None,
    message_costs=None,
    reserve_tokens=0,
):
    """Cache-aware truncation (Character.AI / prompt-poet algorithm).

    Instead of dropping the oldest single turn each call — which moves the
    truncation point every turn and breaks the provider's KV prefix cache — the
    prompt is only touched when it grows past `token_limit`, and then trimmed
    all the way down to `retention_ratio * token_limit` (0.8 by default: keep
    80% of the budget, drop the oldest 20% at once).  The truncation point then
    stays fixed for several turns while the conversation grows back toward the
    limit, giving the GPU prefix cache time to pay off on successive requests.

    Unlike the older `truncation_step` approach (drop a *fixed* token chunk), a
    *variable* number of whole messages is dropped each time and the cut is
    aligned to a safe boundary: it lands on a `user` message or on a plain
    `assistant` message with no `function_call`/`tool_calls`, so a tool call
    (assistant `function_call`/`tool_calls`) and its `function`/`tool`
    results are never separated and no orphaned tool result is left at the
    head of the retained history (strict providers reject that with a 400).
    Allowing plain-assistant boundaries (not just `user` ones) stops a long
    tail of reasoning/function messages from wasting the whole retention
    budget.  The newest user message — the current prompt — is always kept.

    When messages are dropped, a single placeholder `user` message is inserted
    right after the system prompt: "[… N messages omitted from A to B to fit
    context window …]".  The count grows and the trailing timestamp advances on
    each later truncation while the leading timestamp stays fixed, so the model
    knows the history is incomplete and doesn't reason about missing material.
    A `user` role is used (matching the codebase's SYSTEM ALERT convention and
    the guidance for strict-alternation providers, which reject a second
    `system` message) and the real system prompt is left byte-identical so it
    stays cacheable.

    `messages` must NOT include the system message (it is passed separately
    as `system_message`).  Returns the full message list with the system
    message prepended, matching the contract of tokentrim.trim.

    `message_costs` optionally supplies precomputed per-message token counts
    for `messages` (same order), avoiding a second full counting pass for
    callers that already counted — e.g. an incremental conversion cache that
    stashed costs from the previous turn.

    `reserve_tokens` is room to leave for content the caller's runner adds to
    the request *after* trimming — the `execute` tool schema in the
    tool-calling path, or `execution_instructions` appended to the system
    message in the text path.  It is subtracted from `token_limit`, so the
    returned prompt plus those additions still fits the provider's window.

    References:
    - https://github.com/character-ai/prompt-poet#cache-aware-truncation-explained
    - DeepSeek prompt-cache guidance: a retention ratio of 0.8 truncates 20% of
      the context window at once rather than a little bit every time, busting
      the cache far less often.
    """
    if retention_ratio is None:
        retention_ratio = 0.8
    retention_ratio = min(max(retention_ratio, 0.0), 1.0)

    effective_limit = max(token_limit - max(reserve_tokens, 0), 0)
    system_dict = {"role": "system", "content": system_message}
    system_tokens = _count_message_tokens([system_dict], model)
    if message_costs is None:
        message_costs = [_count_message_tokens([m], model) for m in messages]
    total = system_tokens + sum(message_costs)

    if total <= effective_limit:
        return [system_dict] + messages

    # Size the message cut for the retention target, after accounting for the
    # system prompt that `_find_safe_cut` does not see.
    target_tokens = max(effective_limit * retention_ratio - system_tokens, 0)
    cut = _find_safe_cut(messages, target_tokens, model, costs=message_costs)

    trimmed = [system_dict] + messages[cut:]
    if cut > 0:
        trimmed.insert(1, {"role": "user", "content": _omission_note(messages, cut)})

    # The cut above was sized for `target_tokens`, which excludes the system
    # prompt and the omission note added just now.  Recheck the complete
    # request and advance to the next safe boundary until it fits the hard
    # limit — except when only the current prompt is left, which is always
    # kept even if it alone overflows.
    if cut > 0:
        last_user = _last_user_index(messages)
        while _count_message_tokens(trimmed, model) > effective_limit:
            next_cut = _next_safe_cut(messages, cut, last_user)
            if next_cut is None:
                break
            cut = next_cut
            trimmed = [system_dict] + messages[cut:]
            trimmed.insert(
                1, {"role": "user", "content": _omission_note(messages, cut)}
            )
    return trimmed
