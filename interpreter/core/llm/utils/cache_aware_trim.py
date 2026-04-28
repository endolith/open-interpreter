import math

import tiktoken


def _get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _count_message_tokens(messages, model):
    """Approximate token count for a list of OpenAI-format messages.

    Uses the same per-message overhead as OpenAI's cookbook (3 tokens per
    message for role/structure framing, plus 3 tokens for reply priming at
    the end of the list).  Multi-part (vision) content is handled by summing
    the text parts; image tokens are not counted here, so this is an
    undercount when images are present — which is acceptable since
    cache-aware truncation is a heuristic anyway.
    """
    encoding = _get_encoding(model)
    total = 3  # reply priming tokens added by the API
    for msg in messages:
        total += 3  # per-message role/structure overhead
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += len(encoding.encode(content))
        elif isinstance(content, list):
            # Multi-part (vision) content: only count text parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(encoding.encode(part.get("text", "")))
    return total


def cache_aware_trim(messages, system_message, token_limit, truncation_step, model=None):
    """Cache-aware truncation (Character.AI / prompt-poet algorithm).

    Instead of dropping the oldest single turn each call — which moves the
    truncation point every turn and breaks the provider's KV prefix cache —
    we always trim to the nearest multiple of `truncation_step` tokens *below*
    the limit.  The prefix stays identical for roughly
    (truncation_step / avg_tokens_per_turn) consecutive turns before the
    truncation point shifts, giving the provider's GPU prefix cache time to
    pay off on successive requests.

    The trade-off: we occasionally drop slightly more history than strictly
    necessary, in exchange for a significantly higher cache-hit rate and lower
    per-token cost.

    `messages` must NOT include the system message (it is passed separately
    as `system_message`).  Returns the full message list with the system
    message prepended, matching the contract of tokentrim.trim.

    Reference: https://github.com/character-ai/prompt-poet#cache-aware-truncation-explained
    """
    system_dict = {"role": "system", "content": system_message}
    system_tokens = _count_message_tokens([system_dict], model)
    history_tokens = _count_message_tokens(messages, model)
    total = system_tokens + history_tokens

    if total <= token_limit:
        return [system_dict] + messages

    excess = total - token_limit

    # Round excess UP to the nearest truncation_step.  This is the key: we
    # remove slightly more than we must so the truncation point doesn't shift
    # again until the conversation grows by another full truncation_step worth
    # of tokens.
    tokens_to_drop = math.ceil(excess / truncation_step) * truncation_step

    kept = list(messages)
    dropped = 0
    while kept and dropped < tokens_to_drop:
        dropped += _count_message_tokens([kept[0]], model)
        kept.pop(0)

    return [system_dict] + kept
