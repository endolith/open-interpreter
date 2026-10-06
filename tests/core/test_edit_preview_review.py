"""LLM review of every edit dry run, delivered as the edit call's own result.

The workflow: the model calls edit, run_tool_calling_llm dry-runs it and
answers that very tool call with the diff, and respond() then gives the user a
confirmation showing the same diff the model read. A corrected edit earns a
fresh preview (capped per turn); re-emitting the reviewed edit means the model
stands by it and it goes straight to the user.

No fabricated user messages and no verdict tool: the model reads the diff where
a tool result belongs, which keeps the call paired in history and costs exactly
one extra turn. These tests drive respond() with a scripted LLM the way
test_temporary_error_retry.py does, plus a minimal chunk driver standing in for
core._respond_and_store.
"""

import shutil
import unittest

import pytest

import interpreter.core.respond as respond_mod
from interpreter.core.llm.run_tool_calling_llm import _edit_preview_response
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
        self.messages = [{"role": "user", "type": "message", "content": "fix the file"}]
        if edit is not None:
            self.messages.append(dict(edit))
        self.verbose = False
        self.auto_run = auto_run
        self.debug = False
        self.offline = False


def _edit_chunk(language="sed", code="s/a/b/", target="/tmp/target.txt", call_id="call-1"):
    """An assistant edit message shaped like run_tool_calling_llm emits."""
    return {
        "type": "edit",
        "format": language,
        "content": code,
        "target": target,
        "tool_call_id": call_id,
    }


def _preview_response(call_id="call-1", ok=True, output="--- a\n+++ b\n@@"):
    """The dry-run tool response run_tool_calling_llm pairs with the edit call."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "type": "message",
        "content": f"Dry run succeeded. Nothing has been modified yet:\n\n{output}",
        "edit_preview": {"ok": ok, "output": output},
    }


def _drive_to_confirmation(interpreter):
    """Feed respond() chunks back into messages until the confirmation appears.

    Stands in for core._respond_and_store: assistant chunks and paired tool
    responses are stored so the next loop iteration sees them, and the stream is
    closed at the first confirmation (as if the terminal took over prompting).
    """
    confirmations = []
    stream = respond(interpreter)
    try:
        for chunk in stream:
            if chunk.get("type") == "confirmation":
                confirmations.append(chunk)
                break
            if chunk.get("role") == "tool" and chunk.get("type") == "message":
                interpreter.messages.append(dict(chunk))
            elif chunk.get("role") == "assistant" and chunk.get("type") in (
                "message",
                "edit",
            ):
                interpreter.messages.append(dict(chunk))
    finally:
        stream.close()
    return confirmations


def _tool_previews(interpreter):
    """The dry-run results the model was shown this turn."""
    return [m for m in interpreter.messages if isinstance(m.get("edit_preview"), dict)]


@pytest.fixture
def quiet_respond(monkeypatch):
    """Neutralize respond()'s module-level side effects for scripted turns."""
    monkeypatch.setattr(respond_mod, "assemble_system_message", lambda interpreter: "system")
    return monkeypatch


def _proposal(code="s/a/b/", call_id="call-1", ok=True, output="--- a\n+++ b\n@@"):
    """One turn of the real loop: the model calls edit, the dry run answers it.

    The answer can only be read on the next request, which is why every test
    scripts at least one further turn after a proposal.
    """
    return [_edit_chunk(code=code, call_id=call_id), _preview_response(call_id=call_id, ok=ok, output=output)]


def _say(text):
    """A turn where the model only speaks -- its review of the diff."""
    return [{"role": "assistant", "type": "message", "content": text}]


def test_the_model_gets_one_turn_to_read_the_diff_before_the_user_is_asked(quiet_respond):
    """The user is asked only after the model has produced output post-diff.

    The model called edit, the dry run answered that call, and until it says
    anything the user is not asked: that gap is the review. The confirmation
    then shows the very diff the model read -- no second dry run.
    """
    monkeypatch = quiet_respond
    monkeypatch.setattr(
        respond_mod, "dry_run_edit", lambda *a, **k: pytest.fail("must reuse the reviewed diff")
    )
    llm = _FakeLlm([_proposal(), _say("That diff is correct.")])
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 2, "the proposal, then one turn to read the diff"
    assert confirmation["content"]["content"] == "s/a/b/"
    assert confirmation["content"]["dry_run_output"] == "--- a\n+++ b\n@@"


def test_text_streamed_before_the_diff_does_not_count_as_a_review(quiet_respond):
    """Prose written before the preview cannot have been written from it.

    The dry run happens after the tool call returns, so words in the same turn
    as the call predate the diff. The user must not be asked until a later turn
    has actually seen it.
    """
    monkeypatch = quiet_respond
    llm = _FakeLlm(
        [
            [_edit_chunk(), {"role": "assistant", "type": "message", "content": "editing now"}, _preview_response()],
            _say("Now I have seen it."),
        ]
    )
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 2, "the turn that predates the diff is not the review"
    assert confirmation["content"]["content"] == "s/a/b/"


def test_a_corrected_edit_gets_its_own_diff(quiet_respond):
    """Reading the diff and fixing it is the whole point of the review.

    A different edit is a new proposal with its own dry run, and the model reads
    that one before the user sees anything.
    """
    monkeypatch = quiet_respond
    monkeypatch.setattr(
        respond_mod,
        "dry_run_edit",
        lambda language, code, target: {"output": f"diff for {code}", "ok": True},
    )
    llm = _FakeLlm(
        [
            _proposal(code="s/a/b/", call_id="c1", output="diff for s/a/b/"),
            _proposal(code="s/x/y/", call_id="c2", output="diff for s/x/y/"),
            _say("That one is right."),
        ]
    )
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 3, "proposal, correction, then the reviewed correction"
    assert confirmation["content"]["content"] == "s/x/y/"
    assert confirmation["content"]["dry_run_output"] == "diff for s/x/y/"


def test_endless_revisions_land_at_the_user_after_the_cap(quiet_respond):
    """A model that keeps revising still reaches the user, boundedly.

    Each proposal earns one review turn, capped per turn; past the cap the newest
    edit goes to the user with its diff instead of being reviewed forever.
    """
    monkeypatch = quiet_respond
    monkeypatch.setattr(
        respond_mod,
        "dry_run_edit",
        lambda language, code, target: {"output": f"diff {code}", "ok": True},
    )
    cap = respond_mod.MAX_EDIT_REVIEW_ROUNDS
    llm = _FakeLlm([_proposal(code=f"s/a/{i}/", call_id=f"c{i}", output=f"diff s/a/{i}/") for i in range(cap + 2)])
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == cap + 1, f"{cap} review turns, then the user"
    assert confirmation["content"]["content"] == f"s/a/{cap}/", "the newest edit is confirmed"


def test_reemitting_the_reviewed_edit_goes_straight_to_the_user(quiet_respond):
    """The model repeating the same edit means it stands by the diff.

    Repeating it is an answer rather than a new proposal, so it must not buy
    another review turn -- the user gets asked straight away.
    """
    monkeypatch = quiet_respond
    llm = _FakeLlm([_proposal(), _proposal()])
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 2, "the re-emit ends the review instead of restarting it"
    assert confirmation["content"]["content"] == "s/a/b/"


def test_failed_preview_is_reviewed_and_shown_to_the_user(quiet_respond):
    """A dry run that failed is information for the model and for the user.

    It rides the tool result (so the model can fix it) and is attached to the
    confirmation (so the user judges a failure, not a silent change).
    """
    monkeypatch = quiet_respond
    monkeypatch.setattr(
        respond_mod,
        "dry_run_edit",
        lambda language, code, target: {"output": "sed: no commands in code", "ok": False},
    )
    llm = _FakeLlm([_proposal(code="", ok=False, output="sed: no commands in code"), _say("That was broken.")])
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert confirmation["content"]["dry_run_ok"] is False
    assert "no commands" in confirmation["content"]["dry_run_output"]


def test_auto_run_confirms_without_a_review_turn(quiet_respond):
    """auto_run means no dry run happened, so there is nothing to review."""
    monkeypatch = quiet_respond
    monkeypatch.setattr(
        respond_mod, "dry_run_edit", lambda *a, **k: pytest.fail("auto_run must not dry run")
    )
    llm = _FakeLlm([[]])
    interpreter = _FakeInterpreter(llm, _edit_chunk(), auto_run=True)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 0
    assert "dry_run_output" not in confirmation["content"]


def test_edit_keeps_the_provider_tool_call_id_in_converted_history(quiet_respond):
    """The edit call and its preview response must pair in converted history.

    The preview rides the tool response, so if the assistant side were rebuilt
    with a synthesized id instead of the provider's own, the response would
    dangle and thinking-mode providers reject the next request.
    """
    messages = [
        {"role": "user", "type": "message", "content": "edit it"},
        _edit_chunk(call_id="call_abc"),
        _preview_response(call_id="call_abc"),
    ]

    converted = convert_to_openai_messages(
        messages,
        function_calling=True,
        interpreter=_StubConvertInterpreter(),
    )

    assistant = next(m for m in converted if m.get("tool_calls"))
    assert assistant["tool_calls"][0]["id"] == "call_abc"
    tool = next(m for m in converted if m.get("role") == "tool")
    assert tool["tool_call_id"] == "call_abc"


class _StubConvertInterpreter:
    """The interpreter fields convert_to_openai_messages reads off history."""

    user_message_template = "{content}"
    always_apply_user_message_template = False
    shrink_images = True


@unittest.skipUnless(shutil.which("sed"), "sed binary is required for a real dry run")
def test_real_noop_dry_run_sets_no_change(tmp_path):
    """dry_run_edit must flag identical results for the review router.

    The preview text tells the model when its edit would change nothing, so
    this pins the flag at the source: a sed program matching nothing reports ok
    with no_change set, not just a message string.
    """
    target = tmp_path / "file.txt"
    target.write_text("hello\n")

    preview = dry_run_edit("sed", "s/zzz/zzz/", str(target))

    assert preview is not None
    assert preview["ok"] is True
    assert preview.get("no_change") is True