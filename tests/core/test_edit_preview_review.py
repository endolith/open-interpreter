"""LLM review of every edit dry run, via an explicit verdict tool.

The workflow is: the LLM emits an edit, respond() dry-runs it and shows the
diff back, and the model must rule on that diff with the review_edit tool
before it can touch the file again -- while a preview is pending, review_edit
replaces edit in the tool list, so approval cannot be faked by omission or by
prose. approve reaches the user's confirmation prompt; revise feeds the model's
own reason back and previews the corrected edit; no verdict at all means the
user decides unaided. These tests drive respond() with a scripted LLM the way
test_temporary_error_retry.py does, plus a minimal chunk driver that stands in
for core._respond_and_store.
"""

import shutil
import unittest

import pytest

import interpreter.core.respond as respond_mod
from interpreter.core.llm.run_tool_calling_llm import build_request_tools
from interpreter.core.llm.utils.convert_to_openai_messages import (
    convert_to_openai_messages,
)
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
        self.offline = False


class _StubLanguage:
    """Minimal stand-in for a terminal language, for tool-list building."""

    def __init__(self, name):
        self.name = name

    def get_command(self, code=None):
        return f"{self.name} -c"


class _StubTerminal:
    def __init__(self):
        self.languages = [_StubLanguage("python"), _StubLanguage("sh")]


class _StubToolingInterpreter:
    """Just enough interpreter for build_request_tools()."""

    def __init__(self, pending_review=None):
        self.terminal = _StubTerminal()
        self.llm = type("L", (), {"supports_vision": False})()
        if pending_review is not None:
            self._pending_edit_review = pending_review


class _StubConvertInterpreter:
    """The interpreter fields convert_to_openai_messages reads off history."""

    user_message_template = "{content}"
    always_apply_user_message_template = False
    shrink_images = True


def _edit_chunk(language="sed", code="s/a/b/", target="/tmp/target.txt"):
    """An assistant edit message shaped like run_tool_calling_llm emits."""
    return {
        "type": "edit",
        "format": language,
        "content": code,
        "target": target,
    }


def _verdict_chunk(verdict, reason="reviewed"):
    """The marker chunk run_tool_calling_llm yields for a review_edit call."""
    return {
        "type": "edit_review_call",
        "tool_call_id": f"call-{verdict}",
        "verdict": verdict,
        "reason": reason,
    }


def _drive_to_confirmation(interpreter):
    """Feed respond() chunks back into messages until the confirmation appears.

    Stands in for core._respond_and_store: assistant chunks are stored so the
    next loop iteration sees them, verdict markers are stored but not displayed,
    and the stream is closed at the first confirmation (as if the terminal
    interface took over prompting).
    """
    confirmations = []
    stream = respond(interpreter)
    try:
        for chunk in stream:
            if chunk.get("type") == "confirmation":
                confirmations.append(chunk)
                break
            if chunk.get("type") == "edit_review_call":
                interpreter.messages.append(dict(chunk))
            elif chunk.get("role") == "assistant" and chunk.get("type") in (
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


def test_approve_verdict_reaches_the_user_after_one_preview(quiet_respond):
    """The approved diff is the one the user is asked to confirm.

    An edit is previewed, the model approves it with the verdict tool, and only
    then does the confirmation appear -- carrying both the approved edit and
    the dry-run diff the user is being asked to apply.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(monkeypatch, [{"output": "--- a\n+++ b\n@@", "ok": True}] * 5)
    llm = _FakeLlm([[_verdict_chunk("approve", "replaces the one bad line")]])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/b/"))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 1, "the verdict is the one LLM call the review needs"
    assert len(_preview_feedback(interpreter)) == 1
    assert confirmation["format"] == "edit"
    assert confirmation["content"]["content"] == "s/a/b/"
    assert "--- a\n+++ b\n@@" in confirmation["content"]["dry_run_output"]


def test_revise_verdict_feeds_the_reason_back_and_previews_the_fix(quiet_respond):
    """A revise verdict is what drives the retry, and the fix gets its own preview.

    The model's own reason must reach the next turn (it is the actionable part),
    and the corrected edit must be previewed again rather than shown to the user
    unreviewed.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(
        monkeypatch,
        [
            {"output": "--- a\n+++ b\n@@", "ok": True},
            {"output": "--- a\n+++ b\n@@ c", "ok": True},
            {"output": "--- a\n+++ b\n@@ c", "ok": True},
        ],
    )
    # One concern per turn, as the tool list enforces: turn 1 rules "revise",
    # turn 2 emits the corrected edit (only `edit` is available then), turn 3
    # rules on the corrected diff.
    llm = _FakeLlm(
        [
            [_verdict_chunk("revise", "this hits the wrong line")],
            [_edit_chunk(code="s/a/b/")],
            [_verdict_chunk("approve", "now it is the right line")],
        ]
    )
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/x/y/"))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 3, "verdict, revision, verdict: one concern per turn"
    assert len(_preview_feedback(interpreter)) == 2, "both the edit and its fix are previewed"
    feedback = [
        m["content"]
        for m in interpreter.messages
        if m.get("role") == "user" and "asked for a revision" in m.get("content", "")
    ]
    assert len(feedback) == 1
    assert "this hits the wrong line" in feedback[0], "the model's reason must survive"
    assert confirmation["content"]["content"] == "s/a/b/"


def test_endless_revisions_land_at_the_user_after_the_cap(quiet_respond):
    """A model that keeps asking for revisions still lands at the user prompt.

    Each distinct edit earns its own preview round, but rounds are capped per
    turn: past the cap the latest edit falls through to user confirmation with
    its dry-run output attached, unreviewed rather than endlessly re-previewed.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(monkeypatch, [{"output": "--- a\n+++ b\n@@", "ok": True}] * 20)
    cap = respond_mod.MAX_EDIT_PREVIEW_ROUNDS
    llm = _FakeLlm(
        [
            [_verdict_chunk("revise", "try again")],
            [_edit_chunk(code="s/a/b/")],
            [_verdict_chunk("revise", "still wrong")],
            [_edit_chunk(code="s/a/c/")],
        ]
    )
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/x/y/"))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert len(_preview_feedback(interpreter)) == cap
    assert confirmation["content"]["content"] == "s/a/c/"
    assert confirmation["content"]["dry_run_ok"] is True


def test_prose_verdict_is_not_approval_but_does_get_one_ask(quiet_respond):
    """ "The diff is correct -- approve" is not a verdict; ask for the tool call.

    Real models answer the preview in prose, which is exactly why prose must
    never be read as approval. The review is still worth one retry: the model is
    told that words do not count and asked for the review_edit call, and this
    time it rules.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(monkeypatch, [{"output": "--- a\n+++ b\n@@", "ok": True}] * 5)
    llm = _FakeLlm(
        [
            [{"role": "assistant", "type": "message", "content": "The diff is correct. Approve."}],
            [_verdict_chunk("approve", "diff is right")],
        ]
    )
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/b/"))

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 2, "prose earns exactly one ask before the retry"
    asks = [
        m["content"]
        for m in interpreter.messages
        if m.get("role") == "user" and "was not received" in m.get("content", "")
    ]
    assert len(asks) == 1
    assert "Words are not a verdict" in asks[0]
    assert confirmation["content"]["content"] == "s/a/b/"


def test_persistent_silence_reaches_the_user_unapproved(quiet_respond):
    """A model that never rules on the diff hands it to the user, unaided.

    Once the asks are spent there is no verdict and no revision, so the edit
    must reach the user's confirmation with its diff -- not be approved on the
    strength of prose, and not be dropped either.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(monkeypatch, [{"output": "--- a\n+++ b\n@@", "ok": True}] * 20)
    llm = _FakeLlm(
        [
            [{"role": "assistant", "type": "message", "content": "Looks good."}],
            [{"role": "assistant", "type": "message", "content": "Still looks good."}],
            [{"role": "assistant", "type": "message", "content": "Fine by me."}],
        ]
    )
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/b/"))

    (confirmation,) = _drive_to_confirmation(interpreter)

    nags = respond_mod.MAX_EDIT_VERDICT_NAGS
    assert llm.calls == nags + 1, f"one ask per nag, then the user (nags={nags})"
    assert confirmation["content"]["content"] == "s/a/b/", "the edit must not be dropped"
    assert "--- a\n+++ b\n@@" in confirmation["content"]["dry_run_output"]


def test_approval_echo_is_not_rendered_twice(quiet_respond):
    """A model that re-emits the reviewed edit must not print it again.

    Re-emitting the previewed edit is how a model approves it when it does not
    use the verdict tool, and the terminal would otherwise render the identical
    diff a second time right above the confirmation.
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(monkeypatch, [{"output": "--- a\n+++ b\n@@", "ok": True}] * 5)
    llm = _FakeLlm([[_edit_chunk(code="s/a/b/")]])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/b/"))

    confirmations = []
    rendered_edits = []
    stream = respond(interpreter)
    try:
        for chunk in stream:
            if chunk.get("type") == "confirmation":
                confirmations.append(chunk)
                break
            if chunk.get("type") == "edit_review_call":
                interpreter.messages.append(dict(chunk))
            elif chunk.get("role") == "assistant" and chunk.get("type") in ("message", "edit"):
                interpreter.messages.append(dict(chunk))
                if chunk["type"] == "edit":
                    rendered_edits.append(chunk["content"])
    finally:
        stream.close()

    assert rendered_edits == [], "the approval echo must not reach the screen"
    assert len(confirmations) == 1
    assert confirmations[0]["content"]["content"] == "s/a/b/"


def test_review_state_is_cleared_so_no_verdict_tool_leaks_into_later_turns(quiet_respond):
    """The verdict tool is offered only while a review is actually pending.

    build_request_tools() reads interpreter._pending_edit_review from outside
    respond(), so a stale True would hand the model a verdict tool with nothing
    to review (or a replacement `edit` tool it must not have).
    """
    monkeypatch = quiet_respond
    _scripted_dry_run(monkeypatch, [{"output": "--- a\n+++ b\n@@", "ok": True}] * 5)
    llm = _FakeLlm([[_verdict_chunk("approve")]])
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/b/"))

    seen = []
    original = respond_mod.assemble_system_message

    def _record(interpreter_arg):
        seen.append(bool(getattr(interpreter_arg, "_pending_edit_review", None)))
        return original(interpreter_arg)

    monkeypatch.setattr(respond_mod, "assemble_system_message", _record)
    _drive_to_confirmation(interpreter)

    assert seen[0] is False, "no review is pending on the first LLM call"
    assert True in seen, "the review turn sees a pending review"
    assert seen[-1] is False, "the flag is cleared once the edit leaves review"
    assert getattr(interpreter, "_pending_edit_review", None) is None


def test_pending_review_swaps_the_edit_tool_for_the_verdict_tool():
    """While a preview is pending the model cannot edit -- only rule on it.

    Withholding `edit` is what makes the review real: the model is unable to
    revise the file (or silently re-emit it) before it has ruled on the diff, so
    the verdict is the only way forward.
    """
    names = [tool["function"]["name"] for tool in build_request_tools(_StubToolingInterpreter())]
    assert "edit" in names, "the edit tool is offered normally"
    assert "review_edit" not in names

    pending_names = [
        tool["function"]["name"]
        for tool in build_request_tools(_StubToolingInterpreter(pending_review={"format": "sed"}))
    ]
    assert pending_names == ["review_edit"], "a pending preview offers the verdict tool and nothing else"


def test_verdict_marker_converts_to_a_paired_assistant_tool_call():
    """The stored marker must rebuild the tool call for provider history.

    run_tool_calling_llm emits the verdict as a marker plus a role:tool
    acknowledgement. If the marker did not convert back into the assistant
    tool_call it pairs with, process_messages would see an orphan tool response
    and providers would reject the next request -- so the verdict would be lost
    exactly when it matters.
    """
    messages = [
        {"role": "user", "type": "message", "content": "edit it"},
        {
            "type": "edit_review_call",
            "tool_call_id": "call-1",
            "verdict": "approve",
            "reason": "looks right",
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "type": "message",
            "content": "Verdict recorded (approve).",
        },
    ]

    converted = convert_to_openai_messages(
        messages,
        function_calling=True,
        interpreter=_StubConvertInterpreter(),
    )

    assistant = next(m for m in converted if m.get("tool_calls"))
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call-1"
    assert assistant["tool_calls"][0]["function"]["name"] == "review_edit"
    assert '"verdict": "approve"' in assistant["tool_calls"][0]["function"]["arguments"]


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


def test_noop_preview_is_shown_to_the_model_for_a_verdict(quiet_respond):
    """A dry run that changed nothing is previewed like any other edit.

    An edit identical to the file would waste the user's confirmation on a
    no-op, so the no_change flag routes it through review where the model can
    revise it -- or approve a knowingly pointless change.
    """
    monkeypatch = quiet_respond
    no_change = {"output": "sed: no changes (result is identical)", "ok": True, "no_change": True}
    _scripted_dry_run(monkeypatch, [no_change] + [no_change] * 5)
    llm = _FakeLlm(
        [
            [_verdict_chunk("revise", "this is a no-op")],
            [_edit_chunk(code="s/a/a/")],
            [_verdict_chunk("approve", "approving a no-op knowingly")],
        ]
    )
    interpreter = _FakeInterpreter(llm, _edit_chunk(code="s/a/a/"))

    (confirmation,) = _drive_to_confirmation(interpreter)

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
