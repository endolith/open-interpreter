"""
Incremental conversion cache for the per-request LLM history.

Problem: every LLM call rebuilds the request history from the FULL stored
conversation (``respond.py`` re-derives ``messages_for_llm`` each turn) — a
full LMC→OpenAI conversion (re-reading and re-encoding every image from disk)
plus tiktoken counting over the entire history (often ~1M tokens) — before
trimming. For a long conversation that work is repeated on every turn even
though all but the most recent messages are unchanged.

This module stashes the full converted history (plus per-message token costs
and the converter loop's boundary state) after each trim and, when the stored
history has only grown with an unchanged prefix, converts just the appended
suffix and concatenates it onto the stash. The rebuilt list is byte-identical
to a full rebuild, so the request payload — and therefore the trim result and
the provider's prefix-cache behavior — is exactly what it was before. This is
purely a CPU cost saving: it deliberately does NOT change which messages are
sent, where the context is cut, or how stable that cut is. (Cut stability is
``cache_aware_trim``'s behavior: it only trims on overflow, then drops to
``retention_ratio``×limit, so the cut holds for several turns before jumping.
That behavior is unchanged by this cache.)

Only the per-request history sent to the API is affected; the stored JSON
conversation is append-only and untouched.

The stash holds the FULL pre-trim conversion (not the trimmed window), so a
re-trim over it reproduces the exact omission-note counts and timestamps a
full rebuild would. (Re-trimming an already-trimmed window would silently
undercount omitted messages.)

Invalidation — any of these falls back to a full rebuild + restash:
- stored list rebound, shrunk, reordered, or filtered (resume/undo/switch/
  loop-rewrite), or any covered message edited in place (output truncation,
  reasoning replace, code normalization)
- model or any conversion-affecting setting changed
- fresh system message differs (dynamic system rendering)
- image count changed (image-culling interplay), or a covered image file
  changed on disk
- boundary splits a tool-call pair (code with no output yet) or a reasoning
  backfill reach (suffix reasoning + assistant tail)
- current last user message lacks ``sent_at`` (minute-sensitive timestamp)
- text-mode conversion (same-role combining merges across any boundary)
- a custom non-identity user_message_template without always_apply (it
  applies to the last user only, so last-ness changes alter conversion)
- auxiliary title requests (synthetic message list, never stashed)
- loop-mode extra tail appended to the derived list (plus loop mode rebinds
  the stored list every iteration anyway)
- active image culling (4+ images, 2+ in OS mode): the culled derived list no
  longer maps 1:1 onto stored history, so no stash is kept and every call
  fully rebuilds (correct, just uncached)

Known limitation: ``sanitize_messages`` (post-trim, function/tool messages
only) mutates converted dicts in place, so a stash may hold redacted content
from the previous call's sanitize pass. Re-sanitizing is idempotent (no new
secrets found in already-redacted text), so consecutive requests stay
self-consistent and prefix-identical; only the hypothetical full-rebuild
counterfactual could place the cut a few tokens differently, and only when
secrets are present in tool output.
"""

import os

from .cache_aware_trim import _count_message_tokens, cache_aware_trim
from .convert_to_openai_messages import convert_to_openai_messages


def _conversion_settings_key(llm):
    """Values whose change alters conversion output (not just the trim limit)."""
    interp = getattr(llm, "interpreter", None)
    return (
        getattr(llm, "model", None),
        getattr(llm, "supports_functions", None),
        getattr(llm, "supports_vision", None),
        getattr(interp, "shrink_images", None),
        getattr(interp, "user_message_template", None),
        getattr(interp, "always_apply_user_message_template", None),
        getattr(interp, "code_output_template", None),
        getattr(interp, "empty_code_output_template", None),
        getattr(interp, "code_output_sender", None),
    )


def _last_user_has_sent_at(stored):
    """Mirror of ``_user_ts``'s minute-sensitive fallback: a last user message
    without ``sent_at`` gets a timestamp prefix of the current minute, which
    differs across calls and can never be safely reused."""
    for m in reversed(stored):
        if m.get("role") == "user":
            return m.get("sent_at") is not None
    return True


def _is_pairing_clean(stored, covered_len):
    """True when no code/edit in the covered prefix is still waiting for the
    output that would flip its conversion (plain text vs tool_calls).

    Mirrors the pairing forward scan in ``convert_to_openai_messages``: only
    the last code/edit matters, since every earlier one already has a
    non-filtered follower inside the covered prefix.
    """
    for i in range(covered_len - 1, -1, -1):
        m = stored[i]
        if (
            m.get("type") in ("code", "edit")
            and m.get("role") == "assistant"
            and not ("recipient" in m and m["recipient"] != "assistant")
        ):
            for j in range(i + 1, covered_len):
                nx = stored[j]
                if "recipient" in nx and nx["recipient"] != "assistant":
                    continue
                return nx.get("type") == "console" and nx.get("format") == "output"
            # Boundary splits a potential pair: the suffix may bring output.
            return False
    return True


def _suffix_has_reasoning(suffix):
    """Mirror of the converter's reasoning-capture condition."""
    return any(
        m.get("type") == "message"
        and m.get("format") == "reasoning"
        and m.get("role", "assistant") == "assistant"
        and not ("recipient" in m and m["recipient"] != "assistant")
        for m in suffix
    )


def _image_count(messages):
    return sum(1 for m in messages if m.get("type") == "image")


def _path_image_stats(stored, covered_len):
    """File identity per covered file-backed image: conversion re-reads bytes
    from disk, so a changed file must invalidate the stash."""
    stats = []
    for m in stored[:covered_len]:
        if m.get("type") == "image" and m.get("format") == "path":
            try:
                st = os.stat(m.get("content"))
                stats.append((True, st.st_mtime_ns, st.st_size))
            except (OSError, TypeError, ValueError):
                stats.append((False, None, None))
    return stats


class ConversionCache:
    """In-memory stash of one full conversion. Lives on the Llm instance;
    single-threaded use only (same assumption as the rest of the loop)."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.covered_len = 0
        self.snapshot = []
        self.converted = []
        self.costs = []
        self.system_head = None
        self.settings_key = None
        self.seed = {}
        self.image_count = 0
        self.path_stats = []

    def store_full(
        self,
        snapshot,
        converted,
        costs,
        system_head,
        settings_key,
        seed,
        image_count,
        path_stats,
    ):
        self.covered_len = len(snapshot)
        self.snapshot = snapshot
        self.converted = converted
        self.costs = costs
        self.system_head = system_head
        self.settings_key = settings_key
        self.seed = dict(seed)
        self.image_count = image_count
        self.path_stats = path_stats

    def extend(self, stored, snapshot_ext, converted_ext, costs_ext, seed):
        self.covered_len = len(stored)
        self.snapshot.extend(snapshot_ext)
        self.converted.extend(converted_ext)
        self.costs.extend(costs_ext)
        self.seed = dict(seed)


def _fast_path_eligible(cache, stored, derived_bare, system_head, settings_key, interp):
    """All gates for reusing the stashed conversion. Returns the covered
    length on success, 0 when a full rebuild is required."""
    template = getattr(interp, "user_message_template", "{content}")
    if template != "{content}" and not getattr(
        interp, "always_apply_user_message_template", False
    ):
        # A custom template applies to the last user message only, so a
        # covered user that was last at stash time converts differently once
        # a newer prompt arrives. (The default template is the identity, and
        # always_apply is position-independent — both reuse safely.)
        return 0
    covered = cache.covered_len
    if not covered or len(stored) < covered:
        return 0
    if len(derived_bare) != len(stored):
        # Filtered system-role stored messages or an appended loop tail:
        # derived indices no longer map 1:1 onto stored indices.
        return 0
    for i in range(len(stored)):
        if derived_bare[i] is not stored[i]:
            return 0
    if system_head != cache.system_head:
        return 0
    if settings_key != cache.settings_key:
        return 0
    for i in range(covered):
        if stored[i] != cache.snapshot[i]:
            # In-place edit (output truncation, reasoning replace, ...).
            return 0
    if _image_count(stored) != cache.image_count:
        return 0
    if _path_image_stats(stored, covered) != cache.path_stats:
        return 0
    if not _last_user_has_sent_at(stored):
        return 0
    if not _is_pairing_clean(stored, covered):
        return 0
    suffix = stored[covered:]
    if _suffix_has_reasoning(suffix) and (
        not cache.converted or cache.converted[-1].get("role") == "assistant"
    ):
        # A full rebuild would backfill this reasoning into the stashed tail.
        return 0
    return covered


def _convert_suffix(llm, suffix, offset, seed):
    """Convert an appended suffix exactly as a full rebuild would."""
    end_state = {}
    converted = convert_to_openai_messages(
        suffix,
        function_calling=llm.supports_functions,
        vision=llm.supports_vision,
        shrink_images=llm.interpreter.shrink_images,
        interpreter=llm.interpreter,
        stored_offset=offset,
        resume_state=seed,
        end_state=end_state,
    )
    return converted, end_state


def cached_convert_and_trim(llm, derived, token_limit, model, use_cache=True):
    """Drop-in for the convert+split+trim sequence in ``Llm.run``.

    `derived` is run's LMC list (system head + filtered stored history,
    pre-cull). Returns the final trimmed OpenAI list WITH the system message
    prepended (the ``cache_aware_trim`` contract). On any invalidation —
    or when `use_cache` is False (auxiliary title requests) — performs the
    exact legacy full rebuild. Best-effort caching: stash bookkeeping never
    raises (failures skip caching, never the request).
    """
    raw_system_head = derived[0]["content"]
    # Mirror convert_to_openai_messages: string content is stripped.
    system_head = (
        raw_system_head.strip()
        if isinstance(raw_system_head, str)
        else raw_system_head
    )
    derived_bare = derived[1:]
    stored = llm.interpreter.messages
    ratio = llm.retention_ratio
    cache = llm._conversion_cache
    if cache is None:
        cache = ConversionCache()
        llm._conversion_cache = cache

    if (
        use_cache
        and llm.supports_functions
        and isinstance(stored, list)
        and _last_user_has_sent_at(stored)
    ):
        try:
            settings_key = _conversion_settings_key(llm)
            covered = _fast_path_eligible(
                cache,
                stored,
                derived_bare,
                system_head,
                settings_key,
                llm.interpreter,
            )
        except Exception:
            covered = 0
        if covered:
            try:
                suffix = stored[covered:]
                if suffix:
                    suffix_converted, end_state = _convert_suffix(
                        llm, suffix, covered, cache.seed
                    )
                    suffix_costs = [
                        _count_message_tokens([m], model) for m in suffix_converted
                    ]
                else:
                    suffix_converted, suffix_costs = [], []
                    end_state = dict(cache.seed)
                full_converted = cache.converted + suffix_converted
                full_costs = cache.costs + suffix_costs
                out = cache_aware_trim(
                    full_converted,
                    system_head,
                    token_limit,
                    retention_ratio=ratio,
                    model=model,
                    message_costs=full_costs,
                )
                try:
                    cache.extend(
                        stored,
                        [dict(m) for m in suffix],
                        suffix_converted,
                        suffix_costs,
                        end_state,
                    )
                except Exception:
                    pass
                return out
            except Exception:
                pass

    # Full rebuild: image culling + conversion + trim, exactly as llm.run did
    # before (moved here so fast paths can skip it; the culled set provably
    # cannot change while the image-count gate holds). The no-vision renderer
    # branch intentionally stays in llm.run and runs on every call — it is
    # idempotent, and the value-equality gate restashes after it renders.
    image_messages = [msg for msg in derived if msg["type"] == "image"]
    if llm.supports_vision:
        if llm.interpreter.os:
            # Keep only the last two images if the interpreter is running in OS mode
            if len(image_messages) > 1:
                for img_msg in image_messages[:-2]:
                    derived.remove(img_msg)
                    if llm.interpreter.verbose:
                        print("Removing image message!")
        else:
            # Delete all the middle ones (leave only the first and last 2 images) from messages_for_llm
            if len(image_messages) > 3:
                for img_msg in image_messages[1:-2]:
                    derived.remove(img_msg)
                    if llm.interpreter.verbose:
                        print("Removing image message!")
            # Idea: we could set detail: low for the middle messages, instead of deleting them
    # Convert the whole derived list including the system head, exactly like
    # legacy llm.run did (this also preserves the empty-history edge). The
    # -1 offset keeps absolute stored indices: derived position 0 is the
    # system head, so position p maps to stored index p - 1 and synthetic
    # tool-call ids stay index-stable for future suffixes. (When culling
    # removed images the mapping shifts, but then clean_map below fails and
    # nothing is stashed — ids stay request-locally consistent regardless.)
    end_state = {}
    converted = convert_to_openai_messages(
        derived,
        function_calling=llm.supports_functions,
        vision=llm.supports_vision,
        shrink_images=llm.interpreter.shrink_images,
        interpreter=llm.interpreter,
        stored_offset=-1,
        end_state=end_state,
    )
    system_message = converted[0]["content"]
    converted = converted[1:]
    costs = [_count_message_tokens([m], model) for m in converted]
    out = cache_aware_trim(
        converted,
        system_message,
        token_limit,
        retention_ratio=ratio,
        model=model,
        message_costs=costs,
    )
    if use_cache and llm.supports_functions and isinstance(stored, list):
        try:
            clean_map = len(derived) - 1 == len(stored) and all(
                derived[1 + i] is stored[i] for i in range(len(stored))
            )
            if clean_map and _last_user_has_sent_at(stored):
                cache.store_full(
                    [dict(m) for m in stored],
                    converted,
                    costs,
                    system_message,
                    _conversion_settings_key(llm),
                    end_state,
                    _image_count(stored),
                    _path_image_stats(stored, len(stored)),
                )
        except Exception:
            pass
    return out
