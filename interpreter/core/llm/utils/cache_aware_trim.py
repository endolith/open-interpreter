import base64
import math
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

# OpenAI's chat framing: 3 tokens per message for role/structure plus 3 tokens
# of reply priming for the whole request.
_REPLY_PRIMING_TOKENS = 3
_PER_MESSAGE_OVERHEAD_TOKENS = 3

# Model families whose low-detail image cost we know.  Recognising the exact
# name (or its dated snapshot) is deliberate: an alias that merely contains a
# family name, e.g. `some-unknown-gpt-4o-compatible`, may bill differently, so
# it takes the conservative bound instead of the cheap floor.
_IMAGE_MODEL_FAMILIES = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
)

# OpenAI's patch-billed vision models charge per 32x32 patch instead of per
# tile: after fitting the image within 2048 px on its longest side, the cost is
# `patches * multiplier`, capped by a 6144-patch budget.  A fixed low-detail
# floor badly underestimates a large image -- a 1600x1600 image is 2500
# patches, i.e. 4050 tokens on `gpt-4.1-mini`, above the 2833 floor -- so these
# families are charged from the image's actual dimensions.  `detail` does not
# matter here: low, high and auto share the same sizing rules.
_PATCH_SIZE = 32
_PATCH_MAX_DIMENSION = 2048
_PATCH_BUDGET = 6144
# Multipliers are exact hundredths rather than floats: 2,500 patches * 1.62
# is 4,050 tokens in the docs, but the float product is 4050.0000000001 and
# `ceil` would overcharge by a token.  Integer ceiling division is exact.
_PATCHED_IMAGE_MULTIPLIERS = {
    "gpt-4.1-mini": (162, 100),
    "gpt-4.1-nano": (246, 100),
}

# Only the head of a base64 payload is decoded when reading an image header:
# 64k characters is ~49k bytes, which covers PNG/GIF/WebP headers and common
# JPEG layouts.  A JPEG whose frame header falls beyond this slice, or a
# remote URL we cannot inspect, stays on the conservative worst-case path.
_IMAGE_HEADER_CHARS = 64 * 1024
_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)

# Providers whose model names use OpenAI's vision token accounting.  A name
# behind any other provider prefix may bill images differently, so it takes the
# conservative bound rather than the cheap low-detail floor.
_OPENAI_IMAGE_PROVIDERS = ("openai", "azure")


def _get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _model_family(model, known_endpoint=True):
    """The known family a model name belongs to, or `None` when unrecognised.

    A bare name (`gpt-4o`) and a name behind an OpenAI-owned provider prefix
    (`openai/gpt-4o`, `azure/gpt-4o`) are recognised, and a numeric version
    suffix such as `gpt-4o-2024-05-13` stays in the same family.  Any other
    provider prefix (`some-provider/gpt-4o`) may route the name to a server
    with a different image accounting scheme, so it is treated as unknown;
    so is every name when `known_endpoint` is false, i.e. when the caller's
    `api_base` points at a server that may not be the provider the name
    implies.  A suffix that is not version-shaped likewise means we cannot
    assume the family's image accounting.
    """
    if model is None:
        return None
    name = model.strip()
    if not known_endpoint:
        return None
    provider, separator, rest = name.partition("/")
    if separator:
        if provider.lower() not in _OPENAI_IMAGE_PROVIDERS:
            return None
        name = rest
    name = name.lower()
    for family in _IMAGE_MODEL_FAMILIES:
        if name == family:
            return family
        if name.startswith(family + "-") and name[len(family) + 1 :].replace("-", "").isdigit():
            return family
    return None


def _image_dimensions(url):
    """Width and height from a base64 data URL's image header, or `None`.

    OpenAI's patch-billed models need pixel dimensions to bill an image
    correctly.  Reading the header of the payload avoids decoding the whole
    image, which can be megabytes.  Anything whose size cannot be read -- a
    remote URL, an unsupported format, a truncated header -- returns `None` so
    the caller can fall back to the conservative worst case.
    """
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, separator, payload = url.partition(",")
    if not separator or ";base64" not in header.lower():
        return None
    try:
        data = base64.b64decode(payload[:_IMAGE_HEADER_CHARS], validate=False)
    except ValueError:
        return None
    return _header_dimensions(data)


def _header_dimensions(data):
    """Pixel dimensions from a decoded PNG/GIF/WebP/JPEG header, or `None`."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if len(data) >= 10 and data[:6] in (b"GIF87a", b"GIF89a"):
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if (
        len(data) >= 30
        and data[:4] == b"RIFF"
        and data[8:12] == b"WEBP"
        and data[12:16] == b"VP8X"
    ):
        return (
            int.from_bytes(data[24:27], "little") + 1,
            int.from_bytes(data[27:30], "little") + 1,
        )
    if data[:2] == b"\xff\xd8":
        return _jpeg_dimensions(data)
    return None


def _jpeg_dimensions(data):
    """Scan JPEG segments for the frame header that carries width/height."""
    index = 2
    while index + 9 <= len(data):
        if data[index] != 0xFF:
            return None
        marker = data[index + 1]
        index += 2
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            # Standalone markers (TEM, RSTn, SOI, EOI) carry no length field.
            continue
        length = int.from_bytes(data[index : index + 2], "big")
        if length < 2:
            return None
        if marker in _SOF_MARKERS:
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            return width, height
        index += length
    return None


def _patched_image_tokens(width, height, multiplier):
    """Token cost of one patch-billed image; unknown dimensions use the cap.

    The image is fitted within 2048 px on its longest side first (preserving
    aspect ratio and rounding, never enlarging it), then covered in 32x32
    patches.  A 1600x1600 image is 2500 patches, i.e. 4050 tokens on
    `gpt-4.1-mini`; a 2048x2048 worst case is 4096 patches, i.e. 6636 tokens.
    `multiplier` is an exact `(numerator, denominator)` pair so a whole-patch
    cost is not pushed over by floating-point error.
    """
    if width is None or height is None or width <= 0 or height <= 0:
        width = height = _PATCH_MAX_DIMENSION
    longest = max(width, height)
    if longest > _PATCH_MAX_DIMENSION:
        scale = _PATCH_MAX_DIMENSION / longest
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
    patches = math.ceil(width / _PATCH_SIZE) * math.ceil(height / _PATCH_SIZE)
    patches = min(patches, _PATCH_BUDGET)
    numerator, denominator = multiplier
    return -(-patches * numerator // denominator)


def _image_token_estimate(detail, model, known_endpoint=True, image_url=None):
    """Estimated tokens a single image part adds to the request.

    Low-detail images are not uniformly cheap: OpenAI's documented base cost
    for `gpt-4o-mini` is much higher than the 85-token floor that models
    like `gpt-4o` use.  Only a recognised family gets the cheap floor; when the
    model is missing or unknown (possibly another provider, or an alias we
    cannot verify, with a different accounting scheme), assume the
    conservative bound so the trim can never underestimate image cost.
    `known_endpoint` is false when a custom `api_base` may serve the model
    under a name whose image accounting we cannot verify.

    Patch-billed OpenAI models (`gpt-4.1-mini`, `gpt-4.1-nano`) are costed
    from `image_url`'s pixel dimensions, because their price scales with pixel
    area and a fixed floor underestimates a large screenshot.  `detail` is
    ignored for them: low, high and auto all use the same patch rules.
    """
    family = _model_family(model, known_endpoint)
    multiplier = _PATCHED_IMAGE_MULTIPLIERS.get(family)
    if multiplier is not None:
        url = image_url.get("url") if isinstance(image_url, dict) else None
        width, height = _image_dimensions(url) or (None, None)
        return _patched_image_tokens(width, height, multiplier)
    if detail != "low":
        return _IMAGE_TOKENS_DEFAULT
    if family == "gpt-4o-mini":
        return _IMAGE_TOKENS_LOW_DETAIL_CONSERVATIVE
    if family is not None:
        return _IMAGE_TOKENS_LOW_DETAIL
    return _IMAGE_TOKENS_LOW_DETAIL_CONSERVATIVE


def _count_text(obj, encoding, model=None, known_endpoint=True):
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
    (see ``_image_token_estimate``), which is what the API approximately bills;
    patch-billed models additionally read the data URL's header for dimensions.
    """
    if isinstance(obj, str):
        return len(encoding.encode(obj, disallowed_special=()))
    if isinstance(obj, dict):
        if obj.get("type") == "image_url":
            image_url = obj.get("image_url") or {}
            return _image_token_estimate(
                image_url.get("detail"), model, known_endpoint, image_url
            )
        return sum(
            _count_text(value, encoding, model, known_endpoint)
            for value in obj.values()
        )
    if isinstance(obj, list):
        return sum(
            _count_text(item, encoding, model, known_endpoint) for item in obj
        )
    return 0


def _count_message_tokens(messages, model, known_endpoint=True):
    """Approximate token count for a list of OpenAI-format messages.

    Uses the same per-message overhead as OpenAI's cookbook (3 tokens per
    message for role/structure framing, plus 3 tokens for reply priming at
    the end of the list).  Every result therefore already includes the
    request's reply priming, even when it counts a single message — callers
    that aggregate per-message results must charge that priming only once
    (see ``cache_aware_trim`` and ``_find_safe_cut``).

    Image parts are charged a model-aware per-image estimate (see
    ``_count_text``), not their base64 payload size, so images count as a
    bounded amount rather than a huge text blob.
    """
    encoding = _get_encoding(model)
    total = _REPLY_PRIMING_TOKENS  # added once by the API for the whole request
    for msg in messages:
        total += _PER_MESSAGE_OVERHEAD_TOKENS
        total += _count_text(
            {k: v for k, v in msg.items() if k != "role"},
            encoding,
            model,
            known_endpoint,
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


def _find_safe_cut(messages, target_tokens, model, costs=None, known_endpoint=True):
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
    conversion cache — are not charged a second full pass.  Each value is what
    ``_count_message_tokens([message], model)`` returns, so it includes the
    request's reply-priming tokens; this function strips one priming per
    message because the request only pays it once.

    `target_tokens` is a budget for the *messages* only; callers must subtract
    the system prompt and any placeholder they add from the hard limit before
    passing it here (``cache_aware_trim`` does, and then rechecks the complete
    result because the placeholder can push a fitting cut back over the limit).
    """
    if costs is None:
        costs = [
            _count_message_tokens([m], model, known_endpoint) for m in messages
        ]
    # Per-message costs each carry the request's reply priming; the caller
    # already charges it once with the system prompt, so the tail must not
    # multiply it by the number of retained messages.
    retained = sum(costs) - _REPLY_PRIMING_TOKENS * len(costs)

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
        retained -= cost - _REPLY_PRIMING_TOKENS
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
    known_endpoint=True,
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
    stashed costs from the previous turn.  Each value is what
    ``_count_message_tokens([message], model)`` returns; the request's
    reply-priming tokens it contains are charged once, not once per message.

    `known_endpoint` is passed to the image estimator.  Callers set it to
    ``False`` when a custom `api_base` may route the model name to a server
    whose per-image token accounting differs from the provider the name
    implies; unrecognised models then take the conservative bound instead of
    the cheap low-detail floor.

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
    system_tokens = _count_message_tokens([system_dict], model, known_endpoint)
    if message_costs is None:
        message_costs = [
            _count_message_tokens([m], model, known_endpoint) for m in messages
        ]
    # `message_costs` values each include the request's reply priming, which
    # `system_tokens` already accounts for: charge it once, not N times.
    message_tokens = sum(message_costs) - _REPLY_PRIMING_TOKENS * len(message_costs)
    total = system_tokens + message_tokens

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
        while (
            _count_message_tokens(trimmed, model, known_endpoint) > effective_limit
        ):
            next_cut = _next_safe_cut(messages, cut, last_user)
            if next_cut is None:
                break
            cut = next_cut
            trimmed = [system_dict] + messages[cut:]
            trimmed.insert(
                1, {"role": "user", "content": _omission_note(messages, cut)}
            )
    return trimmed
