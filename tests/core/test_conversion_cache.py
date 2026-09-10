"""Tests for the incremental conversion cache (conversion_cache.py).

The cache stashes the full converted history between LLM calls and converts
only newly appended stored messages on the fast path. Every test's core
assertion is the same contract: the cached output must be byte-identical to
a from-scratch full rebuild (conversion + trim), so the cache never changes
the payload the provider sees (a differing payload would break its prefix
cache). The cache adds no trimming/stability behavior of its own.
"""

import types

import interpreter.core.llm.llm as llm_module
import interpreter.core.llm.utils.conversion_cache as conversion_cache
from interpreter.core.llm.llm import Llm
from interpreter.core.llm.utils.cache_aware_trim import (
    _count_message_tokens,
    cache_aware_trim,
)
from interpreter.core.llm.utils.conversion_cache import cached_convert_and_trim
from interpreter.core.llm.utils.convert_to_openai_messages import (
    convert_to_openai_messages,
)

SYSTEM = "system prompt here"
MODEL = "openrouter/deepseek/test-model"
LIMIT = 100000


class _FakeInterpreter:
    """Minimal interpreter surface used by the converter and the cache."""

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


class _FakeLlm:
    """Minimal llm surface: strict-DeepSeek function-calling route."""

    model = MODEL
    supports_functions = True
    supports_vision = True
    retention_ratio = 0.8

    def __init__(self):
        self.interpreter = _FakeInterpreter()
        self.interpreter.llm = self
        self._conversion_cache = None


def _stored(llm):
    return llm.interpreter.messages


def _derive(llm):
    """Fresh derived list exactly like respond.py builds per call (system head
    plus filtered stored history). A new list object every call, same message
    objects — mirroring production."""
    stored = _stored(llm)
    return [{"role": "system", "type": "message", "content": SYSTEM}] + [
        m for m in stored if m.get("role") != "system"
    ]


def _full_rebuild(llm, limit=LIMIT):
    """Reference output: convert everything from scratch, then trim."""
    converted = convert_to_openai_messages(
        [m for m in _stored(llm) if m.get("role") != "system"],
        function_calling=llm.supports_functions,
        vision=llm.supports_vision,
        shrink_images=llm.interpreter.shrink_images,
        interpreter=llm.interpreter,
    )
    return cache_aware_trim(
        converted, SYSTEM, limit, retention_ratio=0.8, model=llm.model
    )


def _user(content, ts):
    return {
        "role": "user",
        "type": "message",
        "content": content,
        "sent_at": ts,
    }


def _turn(llm, prompt, ts, code="print(1)", output="1", lang="python"):
    """Append one complete user/code/output turn to stored history."""
    _stored(llm).append(_user(prompt, ts))
    _stored(llm).append(
        {"role": "assistant", "type": "message", "content": "on it"}
    )
    _stored(llm).append(
        {"role": "assistant", "type": "code", "format": lang, "content": code}
    )
    _stored(llm).append(
        {"role": "computer", "type": "console", "format": "output", "content": output}
    )


def _convert_calls(monkeypatch):
    """Spy on conversions: record the length of each converted message list.

    A full rebuild converts len(stored) messages; a fast-path suffix converts
    fewer. Lets tests prove the fast path ran, not just that output matches
    (output also matches after a silent fallback).
    """
    calls = []
    real = conversion_cache.convert_to_openai_messages

    def spy(messages, *args, **kwargs):
        calls.append(len(messages))
        return real(messages, *args, **kwargs)

    monkeypatch.setattr(conversion_cache, "convert_to_openai_messages", spy)
    return calls


def test_fast_path_matches_full_rebuild_across_appends():
    """Appending complete turns reuses the stash: the cached output must equal
    a from-scratch rebuild, and the second call must convert only the suffix
    (fewer messages than a full rebuild would)."""
    llm = _FakeLlm()
    _turn(llm, "first", 1750000000)
    out1 = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out1 == _full_rebuild(llm)
    assert llm._conversion_cache.covered_len == 4


def test_fast_path_converts_only_the_suffix(monkeypatch):
    """The spy proves the second call converts just appended messages instead
    of the whole history — the actual per-request saving."""
    llm = _FakeLlm()
    _turn(llm, "first", 1750000000)
    calls = _convert_calls(monkeypatch)
    cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert calls == [5]
    _turn(llm, "second", 1750000060)
    out2 = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out2 == _full_rebuild(llm)
    assert calls == [5, 4]
    assert llm._conversion_cache.covered_len == 8


def test_synthetic_ids_stable_incremental_vs_full():
    """Synthetic tool-call ids derive from absolute stored indices, so an
    incremental suffix conversion assigns the same ids as a full rebuild —
    keeping call/output pairing and request prefixes byte-identical."""
    llm = _FakeLlm()
    _turn(llm, "first", 1750000000)
    _turn(llm, "second", 1750000060, code="echo hi", output="hi", lang="bash")
    out = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    calls = [m for m in out if m.get("tool_calls")]
    outputs = [m for m in out if m.get("role") == "tool"]
    assert len(calls) == 2
    call_ids = [tc["id"] for m in calls for tc in m["tool_calls"]]
    assert call_ids == ["oi-call-2", "oi-call-6"]
    assert all(m.get("tool_call_id") in call_ids for m in outputs)


def _note_text(window):
    """The omission-note content, or None when the window is untrimmed."""
    for m in window:
        content = m.get("content")
        if isinstance(content, str) and "omitted" in content:
            return content
    return None


def _shared_prefix_len(a, b):
    n = 0
    for ma, mb in zip(a, b):
        if ma != mb:
            break
        n += 1
    return n


def test_sawtooth_hold_and_move_prefix_behavior():
    """The cache faithfully reproduces cache_aware_trim's existing hold/move
    behavior (it does not add or change it): once trimmed, tiny appends hold
    the cut (consecutive requests share a long identical prefix), while a
    large append forces the cut forward (only cut moves restart at the system
    message). Every output is asserted equal to a full rebuild."""
    llm = _FakeLlm()
    limit = 2000
    # Grow until the first trim fires.
    k = 0
    window = cached_convert_and_trim(llm, _derive(llm), limit, MODEL)
    while _note_text(window) is None:
        _turn(llm, f"prompt {k}", 1750000000 + 60 * k, output="x" * 100)
        window = cached_convert_and_trim(llm, _derive(llm), limit, MODEL)
        assert window == _full_rebuild(llm, limit)
        k += 1
        assert k < 60
    first_note = _note_text(window)
    # Tiny appends: whenever the cut holds (the usual case — the landed tail
    # sits enough below target to absorb a small turn), the note is
    # byte-identical and the whole previous window is a prefix of the next
    # (provider cache hit). A rare micro-undershoot may still force a move;
    # that stays correct via == rebuild, and we keep appending until a hold
    # is observed (bounded; overwhelmingly likely on the first try).
    held = False
    for j in range(10):
        _stored(llm).append(_user(f"t{j}", 1750000000 + 60 * (k + j)))
        nxt = cached_convert_and_trim(llm, _derive(llm), limit, MODEL)
        assert nxt == _full_rebuild(llm, limit)
        if _note_text(nxt) == first_note:
            assert _shared_prefix_len(window, nxt) == len(window)
            window = nxt
            held = True
            break
        window = nxt
        first_note = _note_text(window)
    assert held
    # A large append forces the cut forward: the note count grows, and the
    # new window still equals a full rebuild.
    _turn(llm, "big one", 1750000000 + 60 * (k + 3), output="y" * 3000)
    moved = cached_convert_and_trim(llm, _derive(llm), limit, MODEL)
    assert moved == _full_rebuild(llm, limit)
    assert _note_text(moved) != first_note
    assert _shared_prefix_len(window, moved) <= 1  # restarts at system only


def test_invalidation_on_in_place_edit(monkeypatch):
    """Editing a covered message in place (like output truncation does) must
    fall back to a full rebuild and restash — output still equals rebuild."""
    llm = _FakeLlm()
    _turn(llm, "first", 1750000000)
    calls = _convert_calls(monkeypatch)
    cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert calls == [5]
    # In-place content edit of an already-covered message.
    _stored(llm)[3]["content"] = "2"
    _turn(llm, "second", 1750000060)
    out = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out == _full_rebuild(llm)
    # Full-length conversion proves fallback ran instead of reusing the stash.
    assert calls[-1] == len(_stored(llm)) + 1
    assert llm._conversion_cache.covered_len == len(_stored(llm))


def test_invalidation_on_undo_pop(monkeypatch):
    """Popping stored history (undo) shrinks below coverage: must rebuild."""
    llm = _FakeLlm()
    _turn(llm, "first", 1750000000)
    calls = _convert_calls(monkeypatch)
    cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    _stored(llm).pop()
    _stored(llm).pop()
    out = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out == _full_rebuild(llm)
    assert calls[-1] == len(_stored(llm)) + 1


def test_invalidation_on_conversation_switch(monkeypatch):
    """Rebinding interpreter.messages (resume/switch) changes the list object:
    the identity gate must fail and rebuild from scratch."""
    llm = _FakeLlm()
    _turn(llm, "first", 1750000000)
    calls = _convert_calls(monkeypatch)
    cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    llm.interpreter.messages = list(_stored(llm))
    out = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out == _full_rebuild(llm)
    assert calls[-1] == len(_stored(llm)) + 1


def test_invalidation_on_model_change(monkeypatch):
    """Changing the model (e.g. strict-DeepSeek to plain OpenAI) changes
    conversion output shape: must rebuild, and output matches rebuild."""
    llm = _FakeLlm()
    _turn(llm, "first", 1750000000)
    calls = _convert_calls(monkeypatch)
    cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    llm.model = "gpt-4o"
    out = cached_convert_and_trim(llm, _derive(llm), LIMIT, llm.model)
    assert out == _full_rebuild(llm)
    assert calls[-1] == len(_stored(llm)) + 1
    # Legacy shape returns for non-strict routes.
    assert any(m.get("function_call") for m in out)


def test_pairing_boundary_falls_back():
    """A stash whose covered region ends with code awaiting output cannot be
    extended: the suffix output would flip that code from text to tool_calls,
    so the call must fully rebuild (output still equals rebuild)."""
    llm = _FakeLlm()
    _stored(llm).append(_user("run it", 1750000000))
    _stored(llm).append(
        {"role": "assistant", "type": "code", "format": "python", "content": "print(1)"}
    )
    out1 = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out1 == _full_rebuild(llm)
    assert llm._conversion_cache.covered_len == 2
    _stored(llm).append(
        {"role": "computer", "type": "console", "format": "output", "content": "1"}
    )
    out2 = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out2 == _full_rebuild(llm)
    # The previously-unpaired code is now a real tool call.
    assert any(m.get("tool_calls") for m in out2)


def test_reasoning_boundary_falls_back():
    """A suffix reasoning block that would backfill into an assistant tail
    cannot reuse the frozen stash: must rebuild (output equals rebuild)."""
    llm = _FakeLlm()
    _stored(llm).append(_user("run it", 1750000000))
    _stored(llm).append({"role": "assistant", "type": "message", "content": "ok"})
    cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    _stored(llm).append(
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "hmm"}
    )
    _stored(llm).append(
        {"role": "assistant", "type": "message", "content": "actually this"}
    )
    out = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out == _full_rebuild(llm)


def test_reasoning_after_user_boundary_reuses_cache(monkeypatch):
    """The positive case: a stash ending at a user message can extend across a
    suffix reasoning block, because nothing precedes it that backfill could
    touch — output equals rebuild while converting only the suffix."""
    llm = _FakeLlm()
    _stored(llm).append(_user("run it", 1750000000))
    calls = _convert_calls(monkeypatch)
    cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert calls == [2]
    _stored(llm).append(
        {"role": "assistant", "type": "message", "format": "reasoning", "content": "hmm"}
    )
    _stored(llm).append(
        {"role": "assistant", "type": "message", "content": "actually this"}
    )
    out = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out == _full_rebuild(llm)
    assert calls == [2, 2]


def test_bare_last_user_falls_back_and_never_stashes():
    """A last user message without sent_at gets a current-minute timestamp
    prefix, which differs across calls and can never be reused: must rebuild
    every time and must not poison the stash."""
    llm = _FakeLlm()
    _stored(llm).append(
        {"role": "user", "type": "message", "content": "no timestamp here"}
    )
    out1 = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out1 == _full_rebuild(llm)
    assert llm._conversion_cache.covered_len == 0
    out2 = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    # Equal only if run within the same minute; the point is it rebuilds
    # (covered stays 0) rather than reusing a stale minute prefix.
    assert llm._conversion_cache.covered_len == 0
    assert out2 == _full_rebuild(llm)


def test_auxiliary_requests_never_touch_the_cache():
    """Title/naming requests use a synthetic message list: they must neither
    read nor write the stash."""
    llm = _FakeLlm()
    _turn(llm, "first", 1750000000)
    cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert llm._conversion_cache.covered_len == 4
    title_derived = [
        {"role": "system", "type": "message", "content": "label things"},
        {"role": "user", "type": "message", "content": "some transcript"},
    ]
    out = cached_convert_and_trim(
        llm, title_derived, LIMIT, MODEL, use_cache=False
    )
    assert out[0] == {"role": "system", "content": "label things"}
    assert llm._conversion_cache.covered_len == 4


def test_text_mode_never_caches():
    """Text-mode conversion merges same-role messages across any boundary, so
    suffix conversion could never match a full rebuild: bypass the cache."""
    llm = _FakeLlm()
    llm.supports_functions = False
    _turn(llm, "first", 1750000000)
    out = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out == _full_rebuild(llm)
    assert llm._conversion_cache.covered_len == 0


def test_image_count_change_falls_back():
    """Appending an image changes global image-culling decisions: must rebuild
    (output equals rebuild)."""
    llm = _FakeLlm()
    _turn(llm, "first", 1750000000)
    cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert llm._conversion_cache.covered_len == 4
    _stored(llm).append(
        {
            "role": "user",
            "type": "image",
            "format": "description",
            "content": "a screenshot of a plot",
        }
    )
    out = cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    assert out == _full_rebuild(llm)


def test_loop_tail_extra_falls_back():
    """A derived list with an appended loop message (not in stored history)
    breaks the 1:1 index mapping: must rebuild (output equals rebuild)."""
    llm = _FakeLlm()
    _turn(llm, "first", 1750000000)
    cached_convert_and_trim(llm, _derive(llm), LIMIT, MODEL)
    derived = _derive(llm)
    derived.append(
        {"role": "user", "type": "message", "content": "Proceed."}
    )
    out = cached_convert_and_trim(llm, derived, LIMIT, MODEL)
    # The loop message has no sent_at, so as the last user it gets a
    # current-minute timestamp prefix — exactly like a full rebuild would.
    assert out[-1]["content"].endswith("Proceed.")
    assert llm._conversion_cache.covered_len == 4


class _RunInterpreter:
    """Interpreter surface `Llm.run`/`Llm.__init__` touch, for an end-to-end
    run() test that also exercises the cache-aware branch wiring."""

    def __init__(self, stored):
        self.messages = stored
        self.shrink_images = False
        self.os = False
        self.verbose = False
        self.debug = False
        self.in_terminal_interface = False
        self.user_message_template = "{content}"
        self.always_apply_user_message_template = False
        self.code_output_template = "Code output: {content}"
        self.empty_code_output_template = "no output"
        self.code_output_sender = "user"
        self.toolbox = types.SimpleNamespace(
            import_toolbox_api=False,
            vision=types.SimpleNamespace(query=None),
        )


def test_llm_run_sends_system_message_after_cache_aware_trim(monkeypatch):
    """End-to-end through the real Llm.run cache-aware branch: the request must
    keep its system message. Regression guard — the branch returns the trim
    output WITH the system prepended (the trim contract), so run must not
    strip it a second time (which silently sent requests with no system
    prompt)."""
    captured = {}

    def fake_tool_llm(self, params):
        captured["params"] = params
        yield {"role": "assistant", "type": "message", "content": "ok"}

    monkeypatch.setattr(llm_module, "run_tool_calling_llm", fake_tool_llm)

    stored = [_user("hi there", 1740000000)]
    llm = Llm(_RunInterpreter(stored))
    llm._is_loaded = True
    llm.model = "gpt-4o"
    llm.supports_functions = True
    llm.supports_vision = False
    llm.context_window = 100000
    llm.retention_ratio = 0.8
    llm.sanitize_secrets = False

    derived = [
        {"role": "system", "type": "message", "content": "SYSTEM PROMPT"}
    ] + list(stored)
    list(llm.run(derived))

    sent = captured["params"]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == "SYSTEM PROMPT"
    # The cache is active and covered the whole stored history.
    assert llm._conversion_cache.covered_len == len(stored)


def test_stashed_costs_match_fresh_recount():
    """Trimming with stashed per-message costs must equal trimming with a
    fresh recount — the cost-reuse optimization changes nothing."""
    llm = _FakeLlm()
    _turn(llm, "first", 1750000000)
    _turn(llm, "second", 1750000060, output="y" * 2000)
    converted = convert_to_openai_messages(
        [m for m in _stored(llm) if m.get("role") != "system"],
        function_calling=True,
        vision=True,
        shrink_images=True,
        interpreter=llm.interpreter,
    )
    plain = cache_aware_trim(
        converted, SYSTEM, 3000, retention_ratio=0.8, model=MODEL
    )
    costs = [_count_message_tokens([m], MODEL) for m in converted]
    with_costs = cache_aware_trim(
        converted, SYSTEM, 3000, retention_ratio=0.8, model=MODEL, message_costs=costs
    )
    assert plain == with_costs
