"""LLM-first review of every edit dry run, before the user is prompted.

The workflow is: the LLM emits an edit, respond() shows it the dry-run diff
as feedback and re-invokes it, and only an edit the LLM re-emits unchanged
(approval) reaches the user's confirmation prompt; a different edit earns a
fresh preview, up to a per-turn round cap. These tests drive respond() with a
scripted LLM the way test_temporary_error_retry.py does, plus a minimal chunk
driver that stands in for core._respond_and_store.
"""

import shutil
import unittest

import pytest

import interpreter.core.respond as respond_mod
from interpreter.core.respond import respond
from interpreter.core.tools.file_edit import dry_run_edit


class _FakeLlm:
    """Replays a scripted sequence of run() outcomes, one per LLM invocation."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def run(self, messages):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        yield from outcome


class _FakeInterpreter:
    """The slice of Interpreter that the RUN FILE EDIT path touches."""

    def __init__(self, llm, edit, auto_run=False):
        self.llm = llm
        self.messages = [
            {"role": "user", "type": "message", "content": "fix the file"},
            dict(edit),
        ]
        self.verbose = False
        self.auto_run = auto_run
        self.debug = False


def _edit_chunk(language="sed", code="s/a/b/", target="/tmp/target.txt"):
    """An assistant edit message shaped like run_tool_calling_llm emits."""
    return {
        "type": "edit",
        "format": language,
        "content": code,
        "target": target,
    }


def _drive_to_confirmation(interpreter):
    """Feed respond() chunks back into messages until the confirmation appears.

    Stands in for core._respond_and_store: assistant chunks are stored so the
    next loop iteration sees them, and the stream is closed at the first
    confirmation (as if the terminal interface took over prompting).
    """
    confirmations = []
    stream = respond(interpreter)
    try:
        for chunk in stream:
            if chunk.get("type") == "confirmation":
                confirmations.append(chunk)
                break
            if chunk.get("role") == "assistant" and chunk.get("type") in (
                "message",
                "edit",
            ):
                interpreter.messages.append(dict(chunk))
    finally:
        stream.close()
    return confirmations


def _preview_feedback(interpreter):
    """The dry-run preview messages respond() showed the LLM this turn."""
    return [
        m
        for m in interpreter.messages
        if m.get("role") == "user" and "NOT shown to the user yet" in m.get("content", "")
    ]


@pytest.fixture
def quiet_respond(monkeypatch):
    """Neutralize respond()'s module-level side effects for scripted turns."""
    monkeypatch.setattr(respond_mod, "assemble_system_message", lambda interpreter: "system")
    return monkeypatch


def _scripted_dry_run(monkeypatch, results):
    """Serve dry_run_edit outcomes in order; fail loudly if over-consumed."""
    calls = []

    def _dry_run(language, code, target):
        calls.append((language, code, target))
        return results[len(calls) - 1]

    monkeypatch.setattr(respond_mod, "dry_run_edit", _dry_run)
    return calls


def test_passing_edit_is_previewed_before_user_confirmation(quiet_respond):
    """Even a passing edit goes to the LLM first, not straight to the user.

    The first confirmation the driver sees must come only after an LLM round
    trip over the preview: the confirmation carries the approved edit, and the
    LLM was shown the diff with the user explicitly out of the loop so far.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(monkeypatch, [{"output": "--- a\n+++ b\n@@", "ok": True}] * 5)
    llm = _FakeLlm([[_edit_chunk(code="s/a/b/")]])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/b/"))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 1, "one preview round before any user confirmation"
    assert len(_preview_feedback(interpreter)) == 1
    assert "--- a\n+++ b\n@@" in _preview_feedback(interpreter)[0]["content"]
    assert confirmation["format"] == "edit"
    assert confirmation["content"]["content"] == "s/a/b/"


def test_reemitted_edit_counts_as_approval_not_another_preview(quiet_respond):
    """Re-emitting the previewed edit unchanged approves it.

    Approval must fall through to user confirmation without spending another
    preview round: exactly one LLM call and one preview for an edit the model
    accepts on first sight.
    """
    monkeypatch = quiet_respond
    dry_calls = _scripted_dry_run(monkeypatch, [{"output": "--- a\n+++ b\n@@", "ok": True}] * 5)
    llm = _FakeLlm([[_edit_chunk(code="s/a/b/")]])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/b/"))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 1, "the re-emit approves; it must not trigger a second preview"
    assert len(_preview_feedback(interpreter)) == 1
    assert confirmation["content"]["content"] == "s/a/b/"
    assert len(dry_calls) == 2, "approval re-runs the dry run fresh, never reuses a stale preview"


def test_failing_dry_run_is_revised_before_user_confirmation(quiet_respond):
    """A failing dry run goes back to the LLM with the error, not to the user.

    The user never sees the broken attempt: no confirmation is yielded for it,
    and the confirmation that finally appears carries the revised edit the LLM
    emitted after reading the failure.
    """
    monkeypatch = quiet_respond
    dry_calls = _scripted_dry_run(
        monkeypatch,
        [
            {"output": "sed: no commands in code", "ok": False},
            {"output": "--- a\n+++ b\n@@", "ok": True},
            {"output": "--- a\n+++ b\n@@", "ok": True},
        ],
    )
    llm = _FakeLlm([[_edit_chunk(code="s/a/b/")], [_edit_chunk(code="s/a/b/")]])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code=""))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 2, "preview the broken edit, then preview the revision"
    assert len(dry_calls) == 3
    assert confirmation["content"]["content"] == "s/a/b/"
    assert "sed: no commands in code" in _preview_feedback(interpreter)[0]["content"]


def test_endless_revisions_land_at_the_user_after_the_cap(quiet_respond):
    """A model that revises forever still lands at the user prompt, boundedly.

    Each distinct edit earns its own preview round, but rounds are capped per
    turn: past the cap the latest edit falls through to user confirmation with
    its dry-run output attached for the user to judge.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(
        monkeypatch,
        [{"output": "sed: no commands in code", "ok": False}] * 10,
    )
    cap = respond_mod.MAX_EDIT_PREVIEW_ROUNDS
    llm = _FakeLlm([[_edit_chunk(code="s/a/b/")], [_edit_chunk(code="s/a/c/")]] + [[_edit_chunk(code="s/a/d/")]] * 5)
    interpreter = _FakeInterpreter(llm, _edit_chunk(code=""))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == cap, f"exactly {cap} preview rounds, got {llm.calls}"
    assert confirmation["content"]["content"] == "s/a/c/"
    assert confirmation["content"]["dry_run_ok"] is False


def test_auto_run_skips_preview_and_review(quiet_respond):
    """auto_run means no preview exists, so there is nothing to review.

    With auto_run the dry run is skipped entirely (dry_run_output stays None),
    which must also skip the preview branch and go straight to confirmation.
    """
    monkeypatch = quiet_respond
    dry_calls = _scripted_dry_run(monkeypatch, [])
    llm = _FakeLlm([])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/b/"), auto_run=True)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert dry_calls == [], "auto_run performs no dry run"
    assert llm.calls == 0, "nothing to preview without a dry run"
    assert _preview_feedback(interpreter) == []
    assert "dry_run_output" not in confirmation["content"]


def test_noop_preview_is_shown_before_confirmation(quiet_respond):
    """A dry run that changed nothing is previewed like any other edit.

    An edit identical to the file would waste the user's confirmation on a
    no-op. The no_change flag routes it through the same preview, where the
    LLM can fix it or approve it knowingly.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(
        monkeypatch,
        [
            {
                "output": "sed: no changes (result is identical)",
                "ok": True,
                "no_change": True,
            },
        ]
        + [{"output": "sed: no changes (result is identical)", "ok": True}] * 5,
    )
    llm = _FakeLlm([[_edit_chunk(code="s/a/a/")]])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/a/"))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 1, "a no-op preview still goes to the LLM first"
    assert "changed nothing" in _preview_feedback(interpreter)[0]["content"]
    assert confirmation["content"]["content"] == "s/a/a/"


@unittest.skipUnless(shutil.which("sed"), "sed binary is required for a real dry run")
def test_real_noop_dry_run_sets_no_change(tmp_path):
    """dry_run_edit must flag identical results for the review router.

    The review branch keys off preview["no_change"], so this pins the flag at
    the source: a sed program that matches nothing (output identical to the
    file) must report ok with no_change set, not just a message string.
    """
    target = tmp_path / "file.txt"
    target.write_text("hello\n")

    preview = dry_run_edit("sed", "s/zzz/zzz/", str(target))

    assert preview is not None
    assert preview["ok"] is True
    assert preview.get("no_change") is True, "identical output must set the no_change flag"
