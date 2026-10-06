"""The one-off environment block appended to the system message.

Reports where the run started and when. Shape follows the Codex runtime's system
prompt, which states the same facts in an `<env>` block.

Two properties matter more than the content:

- **It must not move.** The system message is the cached prefix, so a value
  recomputed per turn would churn the cache on every turn and cost real money.
  Both fields come from attributes captured at construction.
- **It must not pretend to be live.** It says where the session *started*, which
  is not where the shell *is* once anything has `cd`'d. The per-command shell
  state line covers the live position; this is the anchor, and conflating the
  two would either go stale or churn.
"""

import os
import re
from datetime import datetime

from interpreter.core.core import OpenInterpreter, _launch_context
from interpreter.core.utils.assemble_system_message import (
    _environment_block,
    assemble_system_message,
)


def _block(interpreter=None):
    return _environment_block(interpreter or OpenInterpreter())


def test_block_reports_the_working_directory():
    """The launch directory is stated, so the model need not spend a turn on pwd."""
    interpreter = OpenInterpreter()
    assert interpreter._launch_cwd == os.getcwd()
    assert interpreter._launch_cwd in _environment_block(interpreter)


def test_block_reports_when_the_conversation_started():
    """The timestamp is present and formatted to the minute."""
    interpreter = OpenInterpreter()
    block = _environment_block(interpreter)
    stamp = interpreter._launch_time.strftime("%Y-%m-%d %H:%M")
    assert stamp in block, block


def test_block_uses_the_env_tag_shape():
    """Wrapped in <env>, matching the Codex runtime's system prompt."""
    block = _block()
    assert "<env>" in block and "</env>" in block
    assert "Here is useful information about the environment" in block


def test_block_is_byte_identical_across_turns():
    """Cache stability: the whole point of capturing it once.

    A block that varied per turn would invalidate the prompt cache on every
    request, which is the failure mode the capture-at-startup design exists to
    avoid.
    """
    interpreter = OpenInterpreter()
    blocks = {_environment_block(interpreter) for _ in range(5)}
    assert len(blocks) == 1, f"block changed between calls: {blocks}"


def test_block_does_not_track_a_changing_cwd():
    """It reports where the run started, not where the shell has moved to.

    The shell's cwd advances as commands `cd`; the block must not follow, or it
    becomes a per-turn value and the cache churn returns.
    """
    interpreter = OpenInterpreter()
    before = _environment_block(interpreter)
    interpreter.terminal.languages[0].cwd = "/somewhere/else"
    assert _environment_block(interpreter) == before


def test_partial_interpreter_gets_no_block():
    """The server, tests and embedders build partial interpreters.

    assemble_system_message is called on objects that never went through
    OpenInterpreter.__init__, so the attributes are fetched defensively rather
    than assumed.
    """

    class _Partial:
        pass

    assert _environment_block(_Partial()) == ""


def test_block_is_present_in_the_assembled_system_message():
    """End to end: it reaches the prompt the model is actually sent."""
    interpreter = OpenInterpreter()
    message = assemble_system_message(interpreter)
    assert "<env>" in message
    assert interpreter._launch_cwd in message


def test_env_block_is_not_duplicated():
    """One block per prompt, however many times assembly runs."""
    interpreter = OpenInterpreter()
    message = assemble_system_message(interpreter)
    assert message.count("<env>") == 1, message.count("<env>")


def test_launch_context_returns_cwd_and_time():
    """The helper exists because __init__'s `os` parameter shadows the module."""
    cwd, started = _launch_context()
    assert cwd == os.getcwd()
    assert isinstance(started, datetime)


def test_block_survives_a_path_containing_template_braces():
    """A cwd with {{...}} must reach the model intact, not be templated away.

    render_message interpolates {{...}} as template variables. The block is
    appended *after* render_message for exactly this reason: interpolated, a
    directory called "/tmp/{{x}}/dir" reaches the model as
    "/tmp/------------------" -- the path is silently destroyed. An earlier
    version of this test asserted only that no braces survived, which passed for
    the wrong reason: the whole path had been mangled.
    """
    interpreter = OpenInterpreter()
    interpreter._launch_cwd = "/tmp/{{not_a_template}}/dir"
    message = assemble_system_message(interpreter)

    assert interpreter._launch_cwd in message, (
        "the launch path must round-trip verbatim, braces included"
    )
    assert "------------------" not in message
