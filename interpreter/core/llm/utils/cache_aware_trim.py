import re

import tiktoken


_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]")


def _get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _count_text(obj, encoding):
    """Recursively count the tokens in every string value of an OpenAI message dict.

    Walks nested dicts/lists so that `content` (string or multi-part),
    `reasoning_content`, `name`, `function_call` and `tool_calls` are all
    counted.  Earlier versions only counted `content`, which massively
    undercounted thinking-model turns: a 20k-character reasoning blob was
    counted as a handful of tokens, so cache-aware trimming never fired.
    """
    if isinstance(obj, str):
        return len(encoding.encode(obj))
    if isinstance(obj, dict):
        return sum(_count_text(value, encoding) for value in obj.values())
    if isinstance(obj, list):
        return sum(_count_text(item, encoding) for item in obj)
    return 0


def _count_message_tokens(messages, model):
    """Approximate token count for a list of OpenAI-format messages.

    Uses the same per-message overhead as OpenAI's cookbook (3 tokens per
    message for role/structure framing, plus 3 tokens for reply priming at
    the end of the list).  Image tokens are not counted, so this is an
    undercount when images are present — which is acceptable since
    cache-aware truncation is a heuristic anyway.
    """
    encoding = _get_encoding(model)
    total = 3  # reply priming tokens added by the API
    for msg in messages:
        total += 3  # per-message role/structure overhead
        total += _count_text({k: v for k, v in msg.items() if k != "role"}, encoding)
    return total


def _find_safe_cut(messages, target_tokens, model):
    """Index of the oldest message to *keep* after dropping the prefix.

    Returns `cut` such that dropping `messages[:cut]` brings the retained tail
    at or below `target_tokens`.  The cut always lands on a `user`-role
    boundary so a tool-call turn (an assistant message holding
    `function_call`/`tool_calls`, immediately followed by its
    `function`/`tool` results) is never split in two, and no orphaned tool
    result is ever left as the first retained message — strict providers
    reject both shapes with a 400.  The newest user message (the current
    prompt) is never dropped, even if it alone exceeds the target.
    """
    costs = [_count_message_tokens([m], model) for m in messages]
    retained = sum(costs)

    last_user = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user = i
            break
    if last_user < 0:
        # No user boundary anywhere: leave the history untouched rather than
        # risk orphaning a tool result.
        return 0

    for i, cost in enumerate(costs):
        if (
            retained <= target_tokens
            and messages[i].get("role") == "user"
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


def cache_aware_trim(messages, system_message, token_limit, retention_ratio=0.8, model=None):
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
    aligned to a natural turn boundary: it always falls on a `user` message, so
    a tool call (assistant `function_call`/`tool_calls`) and its
    `function`/`tool` results are never separated and no orphaned tool result
    is left at the head of the retained history (strict providers reject that
    with a 400).  The newest user message — the current prompt — is always kept.

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

    References:
    - https://github.com/character-ai/prompt-poet#cache-aware-truncation-explained
    - DeepSeek prompt-cache guidance: a retention ratio of 0.8 truncates 20% of
      the context window at once rather than a little bit every time, busting
      the cache far less often.
    """
    if retention_ratio is None:
        retention_ratio = 0.8
    retention_ratio = min(max(retention_ratio, 0.0), 1.0)

    system_dict = {"role": "system", "content": system_message}
    system_tokens = _count_message_tokens([system_dict], model)
    total = system_tokens + _count_message_tokens(messages, model)

    if total <= token_limit:
        return [system_dict] + messages

    target_tokens = token_limit * retention_ratio
    cut = _find_safe_cut(messages, target_tokens, model)

    trimmed = [system_dict] + messages[cut:]
    if cut > 0:
        trimmed.insert(1, {"role": "user", "content": _omission_note(messages, cut)})
    return trimmed
