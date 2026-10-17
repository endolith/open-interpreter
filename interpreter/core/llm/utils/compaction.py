"""LLM-written compaction summaries for cache-aware truncation.

When the conversation outgrows the context window, ``cache_aware_trim`` drops
the oldest messages down to ``retention_ratio`` of the budget.  Dropping them
bare loses everything the session established; this module instead asks the
session's own model to distill the dropped slice into a structured summary
that is appended to the omission note, so the continuing conversation still
knows what came before.

The update rule is anchored and incremental (borrowed from OpenCode's
compaction): each event summarizes *only the newly dropped slice* together
with the previous summary — "preserve still-true details, remove stale ones,
merge in the new facts" — and the result replaces the old summary.  The
sawtooth invariant is unchanged: the note (summary included) only changes at
truncation events, so the request prefix stays byte-identical between them
and the provider's KV prefix cache keeps paying off.

State lives on the ``Llm`` instance as ``llm._compaction`` —
``{"text": str | None, "covered": int}`` — where ``covered`` is the length of
the converted-history prefix the stored text summarizes.  If the state is
missing or inconsistent (restart, history reset, full conversion rebuild),
the next event summarizes its whole dropped slice fresh, anchored on any
previous text: state loss costs one larger summarization call, never
correctness.

Compaction is fail-loud by design: any summarizer failure raises
``CompactionError`` out of the trim path (``Llm.run`` re-raises it past the
trim fallback instead of proceeding untrimmed), the turn aborts with the
history untouched, and the next turn retries the same compaction.  There is
no silent fallback to the plain omission note.
"""


class CompactionError(Exception):
    """The compaction summarizer call failed, refused, or came back unusable.

    Raised instead of sending a request the model was never told is
    truncated.  Carries a human-readable explanation; ``Llm.run`` displays it
    and re-raises so the turn aborts with history untouched.
    """


COMPACTION_SUMMARY_HEADER = "[Summary of the omitted conversation:]"

# Per-string cap when rendering dropped messages for the summarizer.  Console
# output is already truncated upstream, but reasoning blobs and tool results
# can be tens of thousands of characters; the summarizer needs their gist,
# not their bytes.  Long strings are cut with a marker, never silently.
_COMPACTION_STRING_CAP = 4000

# Image payloads must never reach the summarizer as base64: a single data URL
# can dwarf the whole context window.  (Same reasoning as the token counter's
# fixed per-image cost in cache_aware_trim.)
_IMAGE_PLACEHOLDER = "[image omitted from compaction input]"

_COMPACTION_SYSTEM_PROMPT = (
    "You are a context summarization agent. You are given a conversation "
    "between a user and an agent that executes code on the user's machine. "
    "Your goal is a structured summary matching the requested format so the "
    "work can continue from the summary alone."
)

_SUMMARY_TEMPLATE = """Output exactly the Markdown structure shown below and keep the section order unchanged. Write so that the task could continue from this summary alone. Do not continue the task. Do not answer any question in the conversation. Respond in the same language as the conversation.

## Objective
- [what the user asked for; requirements and constraints they stated]

## Environment and state
- [working directory, OS/shell, installed packages, servers or background processes, environment changes — anything later work assumes to be true. Omit if nothing was established.]

## What was done and what happened
- [code executed or commands run and their outcomes — keep exact commands and key results or errors. Terse bullets, not paragraphs.]

## Files touched
- [paths read, created, or edited and what changed. Omit this section if no files were involved — sessions that only run code in memory have none.]

## Decisions and preferences
- [choices made during the work; user-stated preferences to keep honoring]

## Errors and open issues
- [errors seen and whether they were resolved; unresolved blockers]

## Next steps
- [what was in progress when the conversation was summarized; what to do next]

Keep exact file paths, commands, identifiers, and error text when known. Prefer terse bullets over paragraphs."""


def summary_token_cap(token_limit):
    """Max summary output tokens: 5% of the window, clamped to [200, 1500].

    A fixed budget would eat a quarter of a small local window or waste a
    flagship one; scaling keeps the note proportional to the conversation it
    stands in for.  The floor keeps tiny windows usable, the ceiling keeps
    the note from growing into the budget it was meant to free.
    """
    return min(1500, max(200, int(token_limit * 0.05)))


def build_compaction_prompt(previous_summary, max_summary_tokens=None):
    """Assemble the summarizer's user message: anchor + template + budget.

    With a previous summary, the model updates it from the conversation that
    follows (OpenCode's anchored rule); without one, it creates it fresh.
    The token budget is stated in-prompt — like every other compaction
    implementation, this relies on the model honoring a length instruction,
    with max_tokens as the hard backstop — so the note stays proportional
    to the window it stands in for.
    """
    if previous_summary:
        anchor = (
            "A previous summary of the older conversation is below. Update it "
            "using the conversation that follows: preserve details that are "
            "still true, remove anything superseded, and merge in the new "
            "facts. Then output the full updated summary.\n\n"
            f"{previous_summary}"
        )
    else:
        anchor = (
            "Create a new summary of the conversation that follows, covering "
            "everything in it."
        )
    prompt = anchor + "\n\n" + _SUMMARY_TEMPLATE
    if max_summary_tokens:
        prompt += (
            f"\n\nKeep the whole summary under ~{max_summary_tokens} tokens. "
            "Shorter is fine when there is less to preserve."
        )
    return prompt


def _cap_text(text):
    if len(text) > _COMPACTION_STRING_CAP:
        return (
            text[:_COMPACTION_STRING_CAP]
            + f"\n[… {len(text) - _COMPACTION_STRING_CAP} characters omitted "
            "from compaction input …]"
        )
    return text


def _render_content(content):
    """Render converted message content to plain text for the summarizer."""
    if isinstance(content, str):
        return _cap_text(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(_cap_text(str(part)))
                continue
            ptype = part.get("type")
            if ptype == "image_url":
                parts.append(_IMAGE_PLACEHOLDER)
            elif ptype == "text":
                parts.append(_cap_text(part.get("text", "")))
            elif ptype in ("input_audio",):
                parts.append("[audio omitted from compaction input]")
            else:
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(_cap_text(text))
        return "\n".join(p for p in parts if p)
    return _cap_text(str(content))


_ROLE_LABELS = {
    "user": "USER",
    "assistant": "ASSISTANT",
    "function": "TOOL RESULT",
    "tool": "TOOL RESULT",
    "system": "SYSTEM",
}


def render_dropped_transcript(dropped):
    """Render dropped converted messages as role-labeled plain text.

    The summarizer gets strings, not message objects (same shape as
    OpenCode's compaction context): one labeled block per message, tool calls
    named with their arguments, images replaced by a placeholder.  Nothing
    here affects what is sent to the provider for the real turn.
    """
    blocks = []
    for message in dropped:
        role = _ROLE_LABELS.get(message.get("role"), str(message.get("role")).upper())
        lines = [f"### {role}"]
        text = message.get("content")
        if text:
            rendered = _render_content(text)
            if rendered:
                lines.append(rendered)
        function_call = message.get("function_call")
        if function_call:
            lines.append(
                f"called `{function_call.get('name', '?')}`"
                f"({function_call.get('arguments', '')})"
            )
        for tool_call in message.get("tool_calls") or []:
            fn = (tool_call.get("function") or {}) if isinstance(tool_call, dict) else {}
            lines.append(
                f"called `{fn.get('name', '?')}`({fn.get('arguments', '')})"
            )
        name = message.get("name")
        if name and role == "TOOL RESULT":
            lines.append(f"(tool: {name})")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def get_compaction_state(llm):
    """Return the compaction state dict, creating it on first use."""
    state = getattr(llm, "_compaction", None)
    if not isinstance(state, dict):
        state = {"text": None, "covered": 0}
        llm._compaction = state
    state.setdefault("text", None)
    state.setdefault("covered", 0)
    return state


def reset_compaction_coverage(llm):
    """Forget how much history the stored summary covers, keep its text.

    Called whenever the conversion is fully rebuilt (first call, in-place
    edits, settings changes): the old prefix indices no longer map onto the
    new converted list, so the next event summarizes its whole dropped slice
    fresh, anchored on the kept text.
    """
    state = get_compaction_state(llm)
    state["covered"] = 0


def run_compaction_summary(llm, model, transcript, previous_text, max_summary_tokens):
    """Run the summarizer through the session's own model and return its text.

    Goes through ``llm.run`` with the auxiliary-compaction flag so every
    provider quirk (api_base, auth headers, gateway routing, local inference)
    is handled exactly like a normal turn — no parallel param plumbing to
    rot.  Raises ``CompactionError`` on any failure or on an empty summary.
    """
    messages = [
        {
            "role": "system",
            "type": "message",
            "content": _COMPACTION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "type": "message",
            "content": build_compaction_prompt(previous_text, max_summary_tokens)
            + "\n\n---\n\n"
            + transcript,
        },
    ]
    llm._compaction_max_tokens = max_summary_tokens
    try:
        content = ""
        for chunk in llm.run(messages, auxiliary_compaction_request=True):
            if chunk.get("format") == "reasoning":
                continue
            if "content" in chunk:
                content += chunk.get("content") or ""
    except Exception as e:
        raise CompactionError(
            "Compaction failed: the summarizer call raised "
            f"{type(e).__name__}: {e}. The conversation history was left "
            "untouched; the next turn will retry the same compaction. To "
            "turn compaction off, set `compact_history: false` under `llm` "
            "in your profile."
        ) from e
    finally:
        llm._compaction_max_tokens = None
    summary = content.strip()
    if not summary:
        raise CompactionError(
            "Compaction failed: the summarizer returned an empty summary. "
            "The conversation history was left untouched; the next turn "
            "will retry the same compaction. To turn compaction off, set "
            "`compact_history: false` under `llm` in your profile."
        )
    return summary


def make_compaction_summarizer(llm, model, full_converted, token_limit):
    """Build the ``summarize(dropped)`` closure for one trim computation.

    ``dropped`` is the to-be-omitted prefix ``full_converted[:cut]`` handed
    over by ``cache_aware_trim``.  The closure re-compacts incrementally —
    previous summary plus only the newly dropped slice — whenever the stored
    coverage still maps onto this converted list, and summarizes fresh
    (anchored on any previous text) otherwise.  A zero-width new slice
    re-emits the stored summary with no LLM call, keeping the note
    byte-identical so the prefix cache survives.
    """

    def summarize(dropped):
        cut = len(dropped)
        state = get_compaction_state(llm)
        previous_text = state.get("text")
        covered = state.get("covered", 0) or 0
        cap = summary_token_cap(token_limit)
        if previous_text and 0 < covered <= cut <= len(full_converted):
            new_slice = full_converted[covered:cut]
            if not new_slice:
                return previous_text
            transcript = render_dropped_transcript(new_slice)
        else:
            # No usable coverage (first compaction, state lost, or the trim
            # point moved behind the old coverage): summarize the whole
            # dropped slice fresh, anchored on any previous text.
            transcript = render_dropped_transcript(dropped)
        summary = run_compaction_summary(llm, model, transcript, previous_text, cap)
        state["text"] = summary
        state["covered"] = cut
        return summary

    return summarize
