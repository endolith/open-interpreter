"""LLM-first review of failing edit dry runs, before the user is prompted.

When an edit's dry run fails validation (or would change nothing), respond()
appends the dry-run output as feedback and re-invokes the LLM instead of
yielding the confirmation chunk. The user only sees a confirmation once an
edit passes its dry run (or the review budget is spent). These tests drive
respond() with a scripted LLM the way test_temporary_error_retry.py does,
plus a minimal chunk driver that stands in for core._respond_and_store.
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


def test_failing_dry_run_is_revised_before_user_confirmation(quiet_respond):
    """A failing dry run goes back to the LLM, not to the user's prompt.

    The first confirmation the driver sees must already carry the revised
    edit, and the LLM must have been consulted exactly once in between. The
    user never sees the broken attempt: no confirmation is yielded for it.
    """
    monkeypatch = quiet_respond
    dry_calls = _scripted_dry_run(
        monkeypatch,
        [
            {"output": "sed: no commands in code", "ok": False},
            {"output": "--- a\n+++ b\n@@", "ok": True},
        ],
    )
    llm = _FakeLlm([[_edit_chunk(code="s/a/b/")]])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code=""))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 1, "one automatic review round before any confirmation"
    assert len(dry_calls) == 2, "the revised edit gets its own dry run"
    assert confirmation["format"] == "edit"
    assert confirmation["content"]["content"] == "s/a/b/"
    feedback = [
        m
        for m in interpreter.messages
        if m.get("role") == "user" and "has not been shown to the user" in m.get("content", "")
    ]
    assert len(feedback) == 1, "the LLM must be told the user has not seen the broken edit"
    assert "sed: no commands in code" in feedback[0]["content"]


def test_persistently_failing_edit_surfaces_after_the_cap(quiet_respond):
    """An edit that never passes still reaches the user, boundedly.

    The review budget is per exact edit content: an unchanged repeat burns it
    instead of looping forever, and past the cap the confirmation is yielded
    with the failing dry-run output attached for the user to judge.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(
        monkeypatch,
        [{"output": "sed: no commands in code", "ok": False}] * 10,
    )
    cap = respond_mod.MAX_EDIT_PREVIEW_RETRIES
    llm = _FakeLlm([[_edit_chunk(code="")] for _ in range(cap + 5)])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code=""))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == cap, f"exactly {cap} automatic review rounds, got {llm.calls}"
    assert confirmation["content"]["dry_run_ok"] is False
    assert "sed: no commands in code" in confirmation["content"]["dry_run_output"]


def test_revised_edit_gets_a_fresh_review_budget(quiet_respond):
    """The cap counts repeats of one edit, not review rounds overall.

    A genuinely revised edit resets the budget, so a fix that needs two
    attempts of its own is not cut off by the rounds spent on its predecessor.
    Only an edit repeated unchanged past the cap falls through to the user.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(
        monkeypatch,
        [{"output": "sed: no commands in code", "ok": False}] * 10,
    )
    cap = respond_mod.MAX_EDIT_PREVIEW_RETRIES
    revised = [[_edit_chunk(code="s/a/b/")]] * (cap + 5)
    llm = _FakeLlm(revised)
    interpreter = _FakeInterpreter(llm, _edit_chunk(code=""))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == cap + 1, "one round for the original plus a full fresh budget for the revision"
    assert confirmation["content"]["content"] == "s/a/b/"


def test_successful_dry_run_confirms_without_an_extra_llm_turn(quiet_respond):
    """A passing dry run must not cost an LLM round trip.

    Review exists to spare the user broken previews, not to second-guess good
    ones: the confirmation for an edit that already passes is yielded
    immediately, with the LLM never re-invoked.
    """
    monkeypatch = quiet_respond
    dry_calls = _scripted_dry_run(monkeypatch, [{"output": "--- a\n+++ b\n@@", "ok": True}])
    llm = _FakeLlm([])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/b/"))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 0, "no review round for an edit that already passes"
    assert dry_calls[0][1] == "s/a/b/"
    assert confirmation["content"]["content"] == "s/a/b/"


def test_auto_run_skips_dry_run_and_review(quiet_respond):
    """auto_run means no preview exists, so there is nothing to review.

    With auto_run the dry run is skipped entirely (dry_run_output stays None),
    which must also skip the review branch and go straight to confirmation.
    """
    monkeypatch = quiet_respond
    dry_calls = _scripted_dry_run(monkeypatch, [])
    llm = _FakeLlm([])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/b/"), auto_run=True)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert dry_calls == [], "auto_run performs no dry run"
    assert llm.calls == 0, "nothing to review without a preview"
    assert "dry_run_output" not in confirmation["content"]


def test_noop_dry_run_is_sent_back_for_revision(quiet_respond):
    """A dry run that changed nothing is a failure worth revising.

    An edit identical to the file would waste the user's confirmation on a
    no-op. The no_change flag (not a string match) routes it into review,
    where the LLM can either fix it or the cap hands it to the user.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(
        monkeypatch,
        [
            {"output": "sed: no changes (result is identical)", "ok": True, "no_change": True},
            {"output": "--- a\n+++ b\n@@", "ok": True},
        ],
    )
    llm = _FakeLlm([[_edit_chunk(code="s/a/b/")]])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/a/"))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 1, "a no-op preview must trigger a review round"
    assert confirmation["content"]["content"] == "s/a/b/"


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
