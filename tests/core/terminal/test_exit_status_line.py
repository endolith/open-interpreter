"""Exit-status line: a completed shell block that fails says so.

The shell always echoed an end-of-execution marker, but nothing reported the
block's exit status. The only status OI ever surfaced was the rare
"process exited before the end of this block" death message, which fired 0
times across 454 executions in the Terminal-Bench 10x3 run while models
repeatedly had to infer from partial output whether a silent command had
worked (e.g. "Hmm no output? ... odd"). bash now also echoes
``##oi_exit:$?##`` right after the user's block; a zero status is swallowed
as protocol and a non-zero one becomes an ``Exit code: N`` line in the
model's context, before the cwd state line.

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


def _output(language, code):
    """Console output from a run, excluding OI's own state line.

    The state line is added by CwdTrackingMixin after every command, not by
    the command itself; filtering it here keeps assertions about the
    command's own output explicit instead of loosening them.
    """
    return [
        str(chunk["content"])
        for chunk in language.run(code)
        if chunk.get("format") != "active_line"
        and chunk.get("content")
        and not str(chunk["content"]).startswith("[Shell State:")
    ]


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


def test_successful_status_is_swallowed():
    """A zero status is protocol, not output: the model sees nothing."""
    assert Bash().line_postprocessor("##oi_exit:0##\n") is None


def test_failing_status_becomes_exit_code_line():
    """A non-zero status becomes a plain-language line for the model."""
    assert Bash().line_postprocessor("##oi_exit:7##\n") == "Exit code: 7\n"


def test_status_glued_to_newlineless_output_keeps_that_output():
    """printf-style output leaves the shell mid-line, so the echo arrives glued.

    The same no-anchoring rule as the cwd marker: keep what the command
    printed, consume the marker, and only then state the status.
    """
    language = Bash()
    assert (
        language.line_postprocessor("no trailing newline##oi_exit:4##\n")
        == "no trailing newline\nExit code: 4\n"
    )
    assert (
        language.line_postprocessor("no trailing newline##oi_exit:0##\n")
        == "no trailing newline"
    )


def test_malformed_or_unrelated_lines_pass_through():
    """Only our own well-formed echo may be rewritten; nothing else is eaten."""
    language = Bash()
    malformed = "prefix ##oi_exit:zero## suffix\n"
    assert language.line_postprocessor(malformed) == malformed
    plain = "program output\n"
    assert language.line_postprocessor(plain) == plain


def test_failing_block_reports_exit_code(bash):
    """A live bash block whose last command fails reports its status.

    This is the whole feature end to end: protocol echo, marker consumed,
    human line in the stream, and no protocol text left for the model.
    """
    out = _output(bash, "false")
    joined = "".join(out)
    assert "Exit code: 1" in joined
    assert "##oi_exit" not in joined


def test_silent_success_reports_no_status_line(bash):
    """`true` prints nothing and succeeds: no status line, no marker noise."""
    assert _output(bash, "true") == []


def test_normal_output_carries_no_status_noise(bash):
    """Successful blocks with output stay exactly their output."""
    out = "".join(_output(bash, "echo hi"))
    assert out.strip() == "hi"
    assert "Exit code" not in out
    assert "##oi" not in out


def test_failed_command_keeps_stderr_and_adds_status(bash):
    """A real failure keeps its error text and gains the status line.

    Models must see both: the message explains *what* went wrong, the status
    says the block as a whole did not succeed.
    """
    out = "".join(_output(bash, "ls /definitely-not-here-oi-test"))
    assert "No such file or directory" in out
    assert "Exit code: 2" in out
    assert "##oi_exit" not in out
