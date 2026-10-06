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
            if chunk.get("type") == "edit_approved":
                interpreter.messages.append(dict(chunk))
            elif chunk.get("role") == "tool" and chunk.get("type") == "message":
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
    """A turn where the model only speaks. Not an approval."""
    return [{"role": "assistant", "type": "message", "content": text}]


def _approve(reason="diff is right", call_id="approve-1"):
    """The model passing its own gate, plus the acknowledgement answering it."""
    return [
        {
            "type": "edit_approved",
            "tool_call_id": call_id,
            "reason": reason,
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "type": "message",
            "content": "Approved. Nothing has been modified yet.",
        },
    ]


def test_the_user_is_only_asked_after_the_model_approves(quiet_respond):
    """Two gates, in order: the model's, then the user's.

    The model calls edit, reads the dry run, and approves it with approve_edit.
    Only then does the confirmation appear -- carrying the diff the model read,
    with no second dry run.
    """
    monkeypatch = quiet_respond
    monkeypatch.setattr(
        respond_mod, "dry_run_edit", lambda *a, **k: pytest.fail("must reuse the reviewed diff")
    )
    llm = _FakeLlm([_proposal(), _approve()])
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 2, "the proposal, then the turn that rules on it"
    assert confirmation["content"]["content"] == "s/a/b/"
    assert confirmation["content"]["llm_approved"] is True
    assert confirmation["content"]["dry_run_output"] == "--- a\n+++ b\n@@"


def test_words_are_not_an_approval(quiet_respond):
    """Saying the diff looks right does not open the user's gate.

    The model is never asked twice and never nagged, but it is also never asked
    *implicitly*: without approve_edit the edit keeps waiting for the model, and
    only the quiet-turn limit hands it to the user as unapproved.
    """
    monkeypatch = quiet_respond
    quiet = respond_mod.MAX_EDIT_QUIET_TURNS
    llm = _FakeLlm([_proposal()] + [_say("Looks right to me.")] * (quiet + 2))
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert confirmation["content"]["llm_approved"] is False, "prose never approves"
    assert confirmation["content"]["content"] == "s/a/b/", "and the edit is not dropped either"
    assert llm.calls == quiet + 1, f"one ask per quiet turn, then the user (quiet={quiet})"


def test_text_streamed_before_the_diff_does_not_count_as_review(quiet_respond):
    """Prose written before the preview cannot have been written from it.

    The dry run happens after the tool call returns, so words in the same turn
    as the call predate the diff and must not count against the quiet-turn limit.
    """
    monkeypatch = quiet_respond
    quiet = respond_mod.MAX_EDIT_QUIET_TURNS
    llm = _FakeLlm(
        [
            [
                _edit_chunk(),
                {"role": "assistant", "type": "message", "content": "editing now"},
                _preview_response(),
            ]
        ]
        + [_say("thinking out loud")] * (quiet - 1)
        + [_approve()]
    )
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert confirmation["content"]["llm_approved"] is True
    assert confirmation["content"]["content"] == "s/a/b/"


def test_a_corrected_edit_keeps_its_own_diff_and_gate(quiet_respond):
    """Revising is the point: a corrected edit gets its own diff and approval.

    The model rejects the first diff, calls edit again, reads the second, and
    approves that one -- which is the edit the user is asked about.
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
            _approve(call_id="ap-2"),
        ]
    )
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert llm.calls == 3, "proposal, correction, approval"
    assert confirmation["content"]["content"] == "s/x/y/"
    assert confirmation["content"]["dry_run_output"] == "diff for s/x/y/"
    assert confirmation["content"]["llm_approved"] is True


def test_revisions_are_not_capped_but_the_turn_ceiling_is(quiet_respond):
    """A model that keeps fixing its diff keeps getting turns -- up to the ceiling.

    There is no cap on revisions: correcting a diff is the feature working. The
    ceiling is only a runaway guard, and reaching it hands the edit to the user
    flagged as unapproved rather than looping or dropping it.
    """
    monkeypatch = quiet_respond
    monkeypatch.setattr(
        respond_mod,
        "dry_run_edit",
        lambda language, code, target: {"output": f"diff {code}", "ok": True},
    )
    ceiling = respond_mod.MAX_EDIT_REVIEW_TURNS

    def _revision(index):
        code = f"s/a/{index}/"
        return _proposal(code=code, call_id=f"c{index}", output=f"diff {code}")

    llm = _FakeLlm([_revision(i) for i in range(ceiling + 2)])
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    # The ceiling counts requests, and the last of them is the edit that reaches
    # the user -- so `ceiling` calls end with revision number ceiling - 1.
    assert llm.calls == ceiling, f"spends the ceiling ({ceiling}), no more"
    assert confirmation["content"]["llm_approved"] is False
    assert confirmation["content"]["content"] == f"s/a/{ceiling - 1}/", "the newest edit is the user's"


def test_approval_is_positional_not_sticky(quiet_respond):
    """An approval covers the diff that was on screen, not later ones.

    _approval_after() searches forward from the newest edit precisely so a stale
    marker cannot rubber-stamp a revision the model never saw.
    """
    from interpreter.core.respond import _approval_after

    older_approval = {
        "type": "edit_approved",
        "tool_call_id": "ap-1",
        "reason": "fine",
    }
    messages = [
        {"role": "user", "type": "message", "content": "edit"},
        _edit_chunk(code="s/a/b/", call_id="c1"),
        _preview_response(call_id="c1"),
        older_approval,
        _edit_chunk(code="s/x/y/", call_id="c2"),
        _preview_response(call_id="c2"),
    ]
    newest_edit_index = max(i for i, m in enumerate(messages) if m.get("type") == "edit")

    assert _approval_after(messages, newest_edit_index) is None
    assert _approval_after(messages, 1) is older_approval


def test_failed_preview_needs_approval_too(quiet_respond):
    """A failed dry run is no more approvable than a passing one."""
    monkeypatch = quiet_respond
    monkeypatch.setattr(
        respond_mod,
        "dry_run_edit",
        lambda language, code, target: {"output": "sed: no commands in code", "ok": False},
    )
    llm = _FakeLlm([_proposal(code="", ok=False, output="sed: no commands in code"), _approve(reason="intentional")])
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert confirmation["content"]["dry_run_ok"] is False
    assert confirmation["content"]["llm_approved"] is True


def test_the_gate_is_offered_alongside_the_normal_tools():
    """A pending dry run adds approve_edit; it takes nothing away.

    Withholding edit is what made an earlier attempt stall: a model that could
    neither revise nor approve narrated instead. execute stays too, so a model
    that wants to look at the file still can.
    """
    from interpreter.core.llm.run_tool_calling_llm import build_request_tools

    class _Language:
        name = "python"

        def get_command(self, code=None):
            return "python"

    class _Stub:
        def __init__(self, pending):
            self.terminal = type("T", (), {"languages": [_Language()]})()
            self.llm = type("L", (), {"supports_vision": False})()
            if pending:
                self._edit_review_pending = pending

    normal = [t["function"]["name"] for t in build_request_tools(_Stub(False))]
    reviewing = [t["function"]["name"] for t in build_request_tools(_Stub(True))]

    assert "approve_edit" not in normal, "no gate to pass before a preview exists"
    assert "approve_edit" in reviewing
    assert "edit" in reviewing, "the model must still be able to revise"
    assert "execute" in reviewing, "and to look before deciding"


def test_calling_an_unrelated_tool_does_not_approve_or_stall_the_review(quiet_respond):
    """Going off to call another tool is work, not a decision.

    A model that checks something between reading a diff and approving it has
    neither approved nor stalled, so the review waits: the edit stays unapproved
    and the detour does not burn the quiet-turn limit that exists for a model
    that only talks. view_image is used as the detour because respond() declines
    it without reaching for the computer stack.
    """
    monkeypatch = quiet_respond
    quiet = respond_mod.MAX_EDIT_QUIET_TURNS

    def _detour(index):
        """One turn where the model calls a tool that is not the gate."""
        return [
            {"type": "view_image_call", "tool_call_id": f"v{index}", "path": "/tmp/x.png"},
            {
                "role": "tool",
                "tool_call_id": f"v{index}",
                "type": "message",
                "content": "User declined to show image.",
            },
        ]

    # More detour turns than the quiet limit allows. If detours counted as
    # silence the edit would be surfaced unapproved before the gate was reached.
    llm = _FakeLlm([_proposal()] + [_detour(i) for i in range(quiet + 2)] + [_approve()])
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert confirmation["content"]["llm_approved"] is True, "approved when it called the gate"
    assert llm.calls == quiet + 4, f"every detour got its turn ({llm.calls} calls)"


def test_only_silence_counts_against_the_quiet_limit(quiet_respond):
    """A model that answers with words and no call is the case that ends."""
    monkeypatch = quiet_respond
    quiet = respond_mod.MAX_EDIT_QUIET_TURNS
    llm = _FakeLlm([_proposal()] + [_say("still thinking")] * (quiet + 1))
    interpreter = _FakeInterpreter(llm, None)

    (confirmation,) = _drive_to_confirmation(interpreter)

    assert confirmation["content"]["llm_approved"] is False
    assert confirmation["content"]["content"] == "s/a/b/", "surfaced, not dropped"


def test_the_gate_is_reset_between_turns(quiet_respond):
    """A review that ends without confirming must not leave the gate on offer.

    build_request_tools() reads interpreter._edit_review_pending from outside
    this function, so a stale True after a decline would hand the model an
    approve_edit tool with nothing to approve.
    """
    monkeypatch = quiet_respond
    llm = _FakeLlm([_proposal(), _approve()])
    interpreter = _FakeInterpreter(llm, None)

    _drive_to_confirmation(interpreter)

    # The flag is cleared once the edit reaches the user...
    assert interpreter._edit_review_pending is False

    # ...and it starts clean on the next turn even if the review was abandoned.
    interpreter._edit_review_pending = True
    seen = []

    class _Recorder:
        def run(self, messages):
            seen.append(getattr(interpreter, "_edit_review_pending", None))
            return iter(())

    interpreter.llm = _Recorder()
    stream = respond(interpreter)
    try:
        for _ in stream:
            break
    finally:
        stream.close()
    assert seen == [None], "reset before the first request of the turn"


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