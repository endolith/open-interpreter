"""One bracketed status stamp per shell block: outcome first, then location.

A completed bash block reports its exit status and working directory as a
single line -- ``[Exit code: 0 | Shell state: CWD: /app]`` -- because two
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

cmd stays silent for now: ``%ERRORLEVEL%`` is only written by some commands
and survives across blocks, so echoing it would blame a previous block's
failure on the next one (pinned by the cmd test below). PowerShell opts in
with capture-and-reset: ``$LASTEXITCODE`` is zeroed at block start and set
to 1 by the catch, so each block reports only its own outcome.
"""

import shutil

import pytest

from interpreter.core.terminal.languages.bash import Bash
from interpreter.core.terminal.languages.cmd import Cmd
from interpreter.core.terminal.languages.cwd_tracking import CwdTrackingMixin
from interpreter.core.terminal.languages.powershell import (
    PowerShell,
    wrap_in_try_catch,
)
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


@pytest.fixture
def powershell():
    """A live pwsh REPL, terminated even if a test fails."""
    if not shutil.which("pwsh"):
        pytest.skip("pwsh not installed; PowerShell coverage needs it")
    language = PowerShell()
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
            content.startswith("[Shell state:") or content.startswith("[Exit code:")
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


def test_powershell_emits_exit_echo_before_cwd_and_end_markers():
    """PowerShell reads $LASTEXITCODE before the cwd echo and end marker.

    The cwd echo is itself a command, so reading the status after it would
    report that Write-Output instead of the block's last native command.
    """
    ps = object.__new__(PowerShell)
    CwdTrackingMixin.__init__(ps)
    code = ps.preprocess_code("Write-Output hi")
    exit_pos = code.index("##oi_exit:$($LASTEXITCODE)##")
    cwd_pos = code.index("##oi_cwd##")
    end_pos = code.index("##end_of_execution##")
    assert exit_pos < cwd_pos < end_pos


def test_powershell_wrapper_resets_status_and_marks_errors():
    """Every block starts at zero, and a caught error reports 1.

    $LASTEXITCODE survives across blocks, so without the reset a failed native
    command would keep stamping the next, unrelated block.
    """
    wrapped = wrap_in_try_catch("1+1")
    assert "$global:LASTEXITCODE = 0" in wrapped
    assert "$global:LASTEXITCODE = 1" in wrapped


def test_powershell_consumes_ansi_prefixed_exit_marker():
    """pwsh prefixes the real marker with ESC[?1l, which must not hide it."""
    ps = object.__new__(PowerShell)
    CwdTrackingMixin.__init__(ps)
    assert ps.line_postprocessor("\x1b[?1l##oi_exit:3##\n") is None
    assert ps._last_exit_status == 3


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
        f"[Exit code: 0 | Shell state: CWD: {language.cwd}]"
    )


def test_failing_status_stamps_exit_code_first_on_the_state_line():
    """Exit status leads the stamp: how it ended, then where we are."""
    language = Bash()
    assert language.line_postprocessor("##oi_exit:7##\n") is None
    assert language._state_line() == (
        f"[Exit code: 7 | Shell state: CWD: {language.cwd}]"
    )


def test_state_line_without_a_status_has_no_exit_segment():
    """A block whose echo never ran (cmd, death path) keeps the plain stamp."""
    language = Bash()
    assert language._state_line() == f"[Shell state: CWD: {language.cwd}]"


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
    assert "[Exit code: 1 | Shell state: CWD: " in contents
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
    assert contents[0].startswith("[Exit code: 0 | Shell state: CWD: ")


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
    assert "[Exit code: 0 | Shell state: CWD: " in contents


def test_failed_command_keeps_stderr_and_adds_stamp(bash):
    """A real failure keeps its error text and gains the status stamp.

    Models must see both: the message explains *what* went wrong, the stamp
    says the block as a whole did not succeed.
    """
    contents = "".join(_contents(bash, "ls /definitely-not-here-oi-test", keep_meta=True))
    assert "No such file or directory" in contents
    assert "[Exit code: 2 | Shell state: CWD: " in contents
    assert "##oi_exit" not in contents


def test_powershell_success_stamps_exit_zero(powershell):
    """A successful block reports 0 and leaves no protocol text behind."""
    contents = "".join(_contents(powershell, "1+1", keep_meta=True))
    assert "[Exit code: 0 | Shell state: CWD: " in contents
    assert "##oi_exit" not in contents


def test_powershell_caught_error_stamps_one_and_does_not_leak(powershell):
    """A thrown error reports 1, and the next block is zero again.

    The reset between blocks is the point of capture-and-reset: bash's $? is
    rewritten by every command, but $LASTEXITCODE is not.
    """
    failed = "".join(_contents(powershell, 'throw "boom"', keep_meta=True))
    assert "Error: boom" in failed
    assert "[Exit code: 1 | Shell state: CWD: " in failed

    ok = "".join(_contents(powershell, "1+1", keep_meta=True))
    assert "[Exit code: 0 | Shell state: CWD: " in ok
    assert "[Exit code: 1" not in ok
