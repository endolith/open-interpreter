"""Tests for LLM-written compaction summaries (compaction.py + trim hook).

Every test's contract: the omission note's first line is byte-identical with
and without a summary, and each summarizer call receives exactly the
(previous summary, newly dropped slice) pair the incremental scheme promises
— never the whole history twice, never a slice it already covered.  The live
LLM is stubbed: ``run`` yields canned summaries, so these tests pin the
plumbing and boundaries, not model prose.
"""

import pytest

from interpreter.core.llm.utils.cache_aware_trim import (
    _omission_note,
    cache_aware_trim,
)
from interpreter.core.llm.utils.compaction import (
    COMPACTION_SUMMARY_HEADER,
    CompactionError,
    build_compaction_prompt,
    get_compaction_state,
    make_compaction_summarizer,
    render_dropped_transcript,
    run_compaction_summary,
    summary_token_cap,
)
from interpreter.core.llm.utils.conversion_cache import cached_convert_and_trim

SYSTEM = "system prompt here"
MODEL = "openrouter/deepseek/test-model"


class _FakeInterpreter:
    """Minimal interpreter surface used by the converter, cache, and hook."""

    user_message_template = "{content}"
    always_apply_user_message_template = False
    code_output_template = "Code output: {content}"
    empty_code_output_template = "no output"
    code_output_sender = "user"
    shrink_images = True
    verbose = False
    os = False

    def __init__(self):
        self.messages = []
        self.llm = None


class _FakeCompactionLlm:
    """Fake llm with a stubbed streaming ``run`` for the summarizer.

    ``script`` is a queue of canned summaries (strings) or exceptions; each
    ``run`` call consumes one entry and records its input messages.  The
    number of ``run`` calls is the number of summarizer invocations.
    """

    model = MODEL
    supports_functions = True
    supports_vision = True
    retention_ratio = 0.8
    compact_history = True

    def __init__(self, script=()):
        self.interpreter = _FakeInterpreter()
        self.interpreter.llm = self
        self._conversion_cache = None
        self._compaction = None
        self._compaction_max_tokens = None
        self.script = list(script)
        self.run_calls = []

    def run(self, messages, auxiliary_title_request=False, auxiliary_compaction_request=False):
        assert auxiliary_compaction_request, "compaction must use its own aux flag"
        self.run_calls.append(messages)
        item = self.script.pop(0) if self.script else "canned summary"
        if isinstance(item, Exception):
            raise item
        yield {"role": "assistant", "type": "message", "content": item}


def _stored(llm):
    return llm.interpreter.messages


def _derive(llm):
    stored = _stored(llm)
    return [{"role": "system", "type": "message", "content": SYSTEM}] + [
        m for m in stored if m.get("role") != "system"
    ]


def _user(content, ts):
    return {"role": "user", "type": "message", "content": content, "sent_at": ts}


def _turn(llm, prompt, ts, code="print(1)", output="1", lang="python"):
    _stored(llm).append(_user(prompt, ts))
    _stored(llm).append({"role": "assistant", "type": "message", "content": "on it"})
    _stored(llm).append(
        {"role": "assistant", "type": "code", "format": lang, "content": code}
    )
    _stored(llm).append(
        {"role": "computer", "type": "console", "format": "output", "content": output}
    )


def _note_text(window):
    for m in window:
        content = m.get("content")
        if isinstance(content, str) and "omitted" in content:
            return content
    return None


def _grow_until_trim(llm, limit, marker="prompt"):
    """Append turns until the first truncation fires; return that window."""
    k = 0
    window = cached_convert_and_trim(llm, _derive(llm), limit, MODEL)
    while _note_text(window) is None:
        _turn(llm, f"{marker} {k}", 1750000000 + 60 * k, output="x" * 100)
        window = cached_convert_and_trim(llm, _derive(llm), limit, MODEL)
        k += 1
        assert k < 60
    return window


# --- template and budget -----------------------------------------------------


def test_template_covers_task_lifecycle_with_optional_files():
    """The template must match how OI is actually used: arbitrary one-off
    tasks where files are common but never guaranteed, so the files section
    says to omit itself, while commands/outcomes and next steps are
    first-class — and the whole thing reads as a handoff to a continuer."""
    prompt = build_compaction_prompt(None)
    for section in (
        "## Objective",
        "## Environment and state",
        "## What was done and what happened",
        "## Files touched",
        "## Decisions and preferences",
        "## Errors and open issues",
        "## Next steps",
    ):
        assert section in prompt
    assert "Omit this section if no files were involved" in prompt
    assert "Do not continue the task" in prompt


def test_template_anchor_updates_instead_of_restarting():
    """With a previous summary the prompt must carry the anchored update rule
    (preserve still-true, drop superseded, merge new) plus the old text; the
    fresh path must not mention a previous summary."""
    old = "OLD SUMMARY TEXT"
    anchored = build_compaction_prompt(old)
    assert old in anchored
    assert "preserve details that are still true" in anchored
    assert "merge in the new" in anchored
    fresh = build_compaction_prompt(None)
    assert old not in fresh
    assert "Create a new summary" in fresh


def test_summary_token_cap_scales_with_window():
    """The cap is 5% of the window clamped to [200, 1500]: proportional from
    small local windows to flagships, never zero, never window-eating."""
    assert summary_token_cap(8000) == 400
    assert summary_token_cap(100000) == 1500
    assert summary_token_cap(200000) == 1500
    assert summary_token_cap(2000) == 200
    assert summary_token_cap(100) == 200


def test_prompt_states_the_budget():
    """The summarizer is told its token budget in-prompt — the mechanism every
    other implementation relies on instead of a truncation backstop."""
    assert "~400 tokens" in build_compaction_prompt(None, 400)
    assert "tokens" not in build_compaction_prompt(None).split("Keep the whole")[-1]


# --- transcript rendering ----------------------------------------------------


def test_render_labels_roles_and_strips_images():
    """Role-labeled blocks for the summarizer; base64 payloads must never
    reach it (a single data URL could dwarf the window)."""
    dropped = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "on it"},
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "run", "arguments": '{"cmd": "ls"}'},
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "screenshot:"},
                {"type": "image_url", "image_url": {"url": "data:XYZ" * 1000}},
            ],
        },
    ]
    text = render_dropped_transcript(dropped)
    assert "### USER" in text and "do the thing" in text
    assert "called `run`" in text and "ls" in text
    assert "[image omitted from compaction input]" in text
    assert "data:XYZ" not in text


def test_render_caps_giant_strings_with_marker():
    """Reasoning blobs and huge tool outputs are cut with a marker, never
    silently — the summarizer needs gist, not bytes."""
    text = render_dropped_transcript(
        [{"role": "assistant", "content": "r" * 5000}]
    )
    assert len(text) < 5000
    assert "omitted from compaction input" in text


# --- trim hook ---------------------------------------------------------------


def test_note_first_line_identical_with_and_without_summary():
    """The summary must not change the note's first line: counts, timestamps,
    and cache-relevant bytes stay exactly as the plain path produces."""
    messages = [
        {"role": "user", "content": "[2026-08-10 09:00] first"},
        {"role": "assistant", "content": "a" * 500},
        {"role": "user", "content": "[2026-08-10 09:01] second"},
        {"role": "assistant", "content": "b" * 500},
        {"role": "user", "content": "latest prompt here"},
    ]
    plain = cache_aware_trim(messages, SYSTEM, 100, model=MODEL)
    assert _note_text(plain) is not None
    summarized = cache_aware_trim(
        messages, SYSTEM, 100, model=MODEL, summarize=lambda dropped: "S1"
    )
    note = _note_text(summarized)
    assert note.startswith(_note_text(plain).splitlines()[0] + "\n")
    assert COMPACTION_SUMMARY_HEADER in note
    assert note.rstrip().endswith("S1")


def test_omission_note_without_summary_unchanged():
    """Direct unit pin: no summary means the exact historical strings."""
    untimestamped = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "x"},
    ]
    assert _omission_note(untimestamped, 1) == "[… 1 message omitted to fit context window …]"
    assert "Summary" not in _omission_note(untimestamped, 1)
    dated = [
        {"role": "user", "content": "[2026-08-10 09:00] hi"},
        {"role": "assistant", "content": "x"},
    ]
    assert _omission_note(dated, 1) == (
        "[… 1 message omitted from 2026-08-10 09:00 "
        "to 2026-08-10 09:00 to fit context window …]"
    )


# --- end-to-end through cached_convert_and_trim ------------------------------


def test_first_compaction_summarizes_dropped_slice_and_updates_state():
    """The first truncation event summarizes its dropped slice, embeds the
    text under the header, and records coverage for the next event."""
    llm = _FakeCompactionLlm(["summary-one"])
    window = _grow_until_trim(llm, 2000)
    assert llm.run_calls, "expected a summarizer call at the first trim"
    note = _note_text(window)
    assert note is not None and note.rstrip().endswith("summary-one")
    state = get_compaction_state(llm)
    assert state["text"] == "summary-one"
    assert state["covered"] > 0
    # The transcript sent over must contain the oldest dropped content.
    transcript = llm.run_calls[0][1]["content"]
    assert "prompt 0" in transcript
    # Fresh path: no previous summary anchored.
    assert "Create a new summary" in transcript


def test_second_compaction_sends_previous_plus_new_slice_only():
    """The re-compaction rule: previous summary plus ONLY the newly dropped
    messages — the already-covered prefix must not be re-sent."""
    llm = _FakeCompactionLlm(["summary-one", "summary-two"])
    limit = 2000
    _grow_until_trim(llm, limit)
    first_transcript = llm.run_calls[0][1]["content"]
    first_covered = get_compaction_state(llm)["covered"]
    # Force the cut forward with a big turn (same move-the-cut pattern as
    # the sawtooth test).
    _turn(llm, "big one", 1750009999, output="y" * 3000)
    window = cached_convert_and_trim(llm, _derive(llm), limit, MODEL)
    assert _note_text(window).rstrip().endswith("summary-two")
    assert len(llm.run_calls) == 2
    second_prompt = llm.run_calls[1][1]["content"]
    # Anchored on the previous summary...
    assert "summary-one" in second_prompt
    assert "preserve details that are still true" in second_prompt
    # ...and the new slice only: strictly newer than the first slice (the
    # freshly appended big turn itself is retained, not dropped — the slice
    # is the next-oldest band the advancing cut newly covered).
    import re

    first_prompts = {int(n) for n in re.findall(r"prompt (\d+)", first_transcript)}
    second_prompts = {int(n) for n in re.findall(r"prompt (\d+)", second_prompt)}
    assert second_prompts, "expected newly dropped prompts in the second slice"
    assert min(second_prompts) > min(first_prompts)
    assert "prompt 0" not in second_prompt
    assert get_compaction_state(llm)["covered"] > first_covered


def test_unchanged_cut_reemits_summary_with_no_new_call():
    """Between truncation events the note (summary included) must be
    byte-identical — no summarizer call, so the prefix cache survives."""
    llm = _FakeCompactionLlm(["summary-one"])
    limit = 2000
    window = _grow_until_trim(llm, limit)
    calls_after_first = len(llm.run_calls)
    note = _note_text(window)
    # Tiny append that the retention margin absorbs: cut holds.
    _stored(llm).append(_user("tiny", 1750009999))
    nxt = cached_convert_and_trim(llm, _derive(llm), limit, MODEL)
    if _note_text(nxt) == note:
        assert len(llm.run_calls) == calls_after_first
        assert _note_text(nxt).rstrip().endswith("summary-one")


def test_compact_history_off_means_plain_note_and_no_calls():
    """Opt-out is total: the historical note verbatim, zero summarizer calls,
    zero state."""
    llm = _FakeCompactionLlm(["must-never-be-used"])
    llm.compact_history = False
    window = _grow_until_trim(llm, 2000)
    note = _note_text(window)
    assert "Summary of the omitted" not in note
    assert llm.run_calls == []
    assert llm._compaction is None


def test_lost_state_rebuilds_fresh_covering_whole_slice():
    """State loss (restart, reset) degrades gracefully: with no anchor text
    the whole dropped slice is summarized fresh — complete, because the full
    history is still in interpreter.messages, so nothing is lost, only the
    incremental efficiency."""
    llm = _FakeCompactionLlm(["summary-one", "summary-fresh"])
    limit = 2000
    _grow_until_trim(llm, limit)
    # Simulate state loss with history intact.
    llm._compaction = None
    llm._conversion_cache = None  # force the full-rebuild path too
    _turn(llm, "after restart", 1750009999, output="z" * 3000)
    window = cached_convert_and_trim(llm, _derive(llm), limit, MODEL)
    assert _note_text(window).rstrip().endswith("summary-fresh")
    prompt = llm.run_calls[-1][1]["content"]
    assert "Create a new summary" in prompt  # no anchor survived
    assert "prompt 0" in prompt  # whole slice re-covered, not just the delta


def test_full_rebuild_after_edit_keeps_text_resets_coverage():
    """An in-place edit invalidates converted-prefix indices: coverage
    resets so the next event re-covers its whole slice (anchored), instead
    of slicing from a stale offset into shifted content."""
    llm = _FakeCompactionLlm(["summary-one", "summary-two"])
    limit = 2000
    _grow_until_trim(llm, limit)
    assert get_compaction_state(llm)["covered"] > 0
    # In-place edit of covered history forces a full rebuild next call.
    _stored(llm)[1]["content"] = "edited"
    _turn(llm, "post edit", 1750009999, output="w" * 3000)
    window = cached_convert_and_trim(llm, _derive(llm), limit, MODEL)
    assert _note_text(window) is not None
    if len(llm.run_calls) == 2:
        prompt = llm.run_calls[1][1]["content"]
        assert "summary-one" in prompt  # text survives; only coverage reset
        assert "prompt 0" in prompt  # whole slice re-covered from the start


# --- failure semantics -------------------------------------------------------


def test_empty_summary_is_a_loud_failure():
    """An empty summarizer result must abort, not ship an empty note."""
    llm = _FakeCompactionLlm(["   "])
    with pytest.raises(CompactionError):
        _grow_until_trim(llm, 2000)


def test_summarizer_exception_is_a_loud_failure_with_guidance():
    """Provider errors surface as CompactionError naming the cause, the
    untouched history, the next-turn retry, and the opt-out."""
    llm = _FakeCompactionLlm([RuntimeError("boom 429")])
    with pytest.raises(CompactionError) as exc_info:
        _grow_until_trim(llm, 2000)
    message = str(exc_info.value)
    assert "429" in message
    assert "untouched" in message
    assert "retry" in message
    assert "compact_history" in message


def test_failed_compaction_leaves_no_state_behind():
    """A failed event must not record coverage: the next turn retries the
    same slice instead of building on a summary that was never made."""
    llm = _FakeCompactionLlm([RuntimeError("nope"), "summary-ok"])
    limit = 2000
    try:
        _grow_until_trim(llm, limit)
    except CompactionError:
        pass
    assert get_compaction_state(llm)["text"] is None
    assert get_compaction_state(llm)["covered"] == 0


# --- closure-level fast path -------------------------------------------------


def test_zero_width_slice_reemits_without_call():
    """cut == covered means nothing new was dropped: return the stored text
    with no LLM call, keeping the note byte-identical."""
    llm = _FakeCompactionLlm(["never-used"])
    llm._compaction = {"text": "STORED", "covered": 3}
    full = [{"role": "user", "content": str(i)} for i in range(6)]
    summarize = make_compaction_summarizer(llm, MODEL, full, 100000)
    assert summarize(full[:3]) == "STORED"
    assert llm.run_calls == []
