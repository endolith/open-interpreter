"""One bracketed status stamp per shell block: outcome first, then location.

A completed bash block reports its exit status and working directory as a
single line -- ``[Exit code: 0 | Shell State: CWD: /app]`` -- because two
separate bracketed stamps read as two unrelated annotations, and an
unbracketed status read as program output. Exit status comes first: "how
that ended, and here's where we are now" (cf. #401, which asks for the same
kind of runtime evidence while a command is still running).

The ``##oi_exit:$?##`` echo is consumed into a stash by
``_filter_exit_marker``; the value rides into ``_state_line`` at the end of
the block. Always reports, success included: silence after a run leaves the
model guessing whether the block worked at all -- the confusion behind the
silent-success cases ("Hmm no output? ... odd") in the Terminal-Bench
transcripts.

cmd and PowerShell stay silent for now: their status variables
(``%ERRORLEVEL%``/``$LASTEXITCODE``) are only written by some commands and
survive across blocks, so echoing them would blame a previous block's
failure on the next one (pinned by the cmd test below).
"""

import pytest

from interpreter.core.terminal.languages.bash import Bash
from interpreter.core.terminal.languages.cmd import Cmd
from interpreter.core.terminal.languages.cwd_tracking import CwdTrackingMixin
from tests.helpers import require_bash_compatible_shell


@pytest.fixture
def bash():
    """A live bash REPL, terminated even if a test fails."""
    require_bash_compatible_shell()
    language = Bash()
    language.start_process()
    try:
        yield language
    finally:
        language.terminate()


def _contents(language, code, keep_meta=False):
    """Console contents from a run.

    OI's own status stamp (and the state line) is excluded unless
    ``keep_meta``: the purity tests are about what the *command* output.
    """
    out = []
    for chunk in language.run(code):
        if chunk.get("format") == "active_line" or not chunk.get("content"):
            continue
        content = str(chunk["content"])
        if keep_meta or not (
            content.startswith("[Shell State:") or content.startswith("[Exit code:")
        ):
            out.append(content)
    return out


def test_bash_emits_exit_echo_before_cwd_and_end_markers():
    """The status echo must run before the cwd echo, which would clobber $?."""
    code = Bash().preprocess_code("true")
    exit_pos = code.index("##oi_exit:$?##")
    cwd_pos = code.index("##oi_cwd##")
    end_pos = code.index("##end_of_execution##")
    assert exit_pos < cwd_pos < end_pos


def test_cmd_stays_silent_until_errorlevel_is_reliable():
    """cmd keeps today's behavior: no exit echo, no false per-block statuses.

    %ERRORLEVEL% survives across blocks and is not written by every command,
    so echoing it would attribute a stale failure to a successful block. The
    day cmd gets capture-and-reset wiring, this test should be replaced by
    cmd coverage of its own.
    """
    # Cmd.__init__ refuses to build off Windows, but preprocessing is pure
    # string work: construct without the gate so the pin runs everywhere.
    cmd = object.__new__(Cmd)
    CwdTrackingMixin.__init__(cmd)
    code = cmd.preprocess_code("echo hi")
    assert "##oi_exit" not in code


def test_successful_block_stamps_exit_zero_on_the_state_line():
    """A zero status is reported too -- silent success was the ambiguous case.

    The marker line itself is consumed (not re-emitted); the value appears
    once, on the state line, so there is exactly one bracketed stamp.
    """
    language = Bash()
    assert language.line_postprocessor("##oi_exit:0##\n") is None
    assert language._state_line() == (
        f"[Exit code: 0 | Shell State: CWD: {language.cwd}]"
    )


def test_failing_status_stamps_exit_code_first_on_the_state_line():
    """Exit status leads the stamp: how it ended, then where we are."""
    language = Bash()
    assert language.line_postprocessor("##oi_exit:7##\n") is None
    assert language._state_line() == (
        f"[Exit code: 7 | Shell State: CWD: {language.cwd}]"
    )


def test_state_line_without_a_status_has_no_exit_segment():
    """A block whose echo never ran (cmd, death path) keeps the plain stamp."""
    language = Bash()
    assert language._state_line() == f"[Shell State: CWD: {language.cwd}]"


def test_status_glued_to_newlineless_output_keeps_that_output():
    """printf-style output leaves the shell mid-line, so the echo arrives glued.

    The same no-anchoring rule as the cwd marker: keep what the command
    printed, consume the marker, and stash the status for the state line --
    for a failure and for a success alike.
    """
    language = Bash()
    assert (
        language.line_postprocessor("no trailing newline##oi_exit:4##\n")
        == "no trailing newline"
    )
    assert language._last_exit_status == 4
    language.line_postprocessor("##oi_exit:0##\n")
    assert language._last_exit_status == 0


def test_malformed_or_unrelated_lines_pass_through():
    """Only our own well-formed echo may be consumed; nothing else is eaten."""
    language = Bash()
    malformed = "prefix ##oi_exit:zero## suffix\n"
    assert language.line_postprocessor(malformed) == malformed
    plain = "program output\n"
    assert language.line_postprocessor(plain) == plain
    assert language._last_exit_status is None


def test_failing_block_stamps_combined_line(bash):
    """A live bash block whose last command fails shows one bracketed stamp.

    This is the whole feature end to end: protocol echo consumed, marker
    gone, outcome and cwd in a single line, and no protocol text left for
    the model.
    """
    contents = "".join(_contents(bash, "false", keep_meta=True))
    assert "[Exit code: 1 | Shell State: CWD: " in contents
    assert "##oi_exit" not in contents
    # Exactly one stamp, not a second bracketed exit line.
    assert contents.count("[Exit code:") == 1


def test_silent_success_stamps_exit_zero(bash):
    """`true` prints nothing: the explicit zero is what tells it worked.

    Without this line the model sees the same nothing it saw when a silent
    command failed -- the exact confusion the feature removes.
    """
    contents = _contents(bash, "true", keep_meta=True)
    assert len(contents) == 1
    assert contents[0].startswith("[Exit code: 0 | Shell State: CWD: ")


def test_normal_output_keeps_output_and_adds_one_stamp(bash):
    """Successful blocks keep their output and gain exactly one stamp."""
    contents = _contents(bash, "echo hi", keep_meta=True)
    assert any(c.strip() == "hi" for c in contents)
    assert sum(1 for c in contents if c.startswith("[Exit code:")) == 1
    assert "##oi" not in "".join(contents)


def test_last_command_status_wins(bash):
    """Pins bash semantics: $? is the block's last command, not a verdict on every line.

    A failure mid-block that prints nothing is only visible through its own
    stderr; the stamp reports the block's overall outcome. Documented so
    nobody mistakes it for per-command accounting.
    """
    contents = "".join(_contents(bash, "false; echo done", keep_meta=True))
    assert "done" in contents
    assert "[Exit code: 0 | Shell State: CWD: " in contents


def test_failed_command_keeps_stderr_and_adds_stamp(bash):
    """A real failure keeps its error text and gains the status stamp.

    Models must see both: the message explains *what* went wrong, the stamp
    says the block as a whole did not succeed.
    """
    contents = "".join(_contents(bash, "ls /definitely-not-here-oi-test", keep_meta=True))
    assert "No such file or directory" in contents
    assert "[Exit code: 2 | Shell State: CWD: " in contents
    assert "##oi_exit" not in contents
