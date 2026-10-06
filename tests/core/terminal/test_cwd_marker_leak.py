"""Regression tests for the cwd-probe marker leaking into program output.

A command whose output does not end in a newline leaves the shell mid-line when
the marker echo runs, so the marker arrives glued to the end of that output:

    no trailing newline##oi_cwd##/home/user/project

The parser used to anchor on ``^``, so it missed that line entirely. Both the
marker and the absolute path then reached the model's context and the user's
display -- and because the marker was named ``##oi_pwd##``, in the same
##tag## shape credential masking uses, a leak of it looked exactly like a leaked
password and cost a debugging session chasing a credential that was never there.
"""

import os
import shutil

import pytest

from interpreter.core.terminal.languages.bash import Bash

# These drive a live bash. A non-bash $SHELL (fish, csh) hangs rather than
# fails on bash syntax, so skip with a reason instead. This branch has no
# tests/helpers.py, so the check is local.
_BASH = shutil.which("bash")
pytestmark = [
    pytest.mark.skipif(not _BASH, reason="bash not installed"),
    pytest.mark.skipif(
        not os.environ.get("SHELL", "bash").endswith(("bash", "zsh")),
        reason="$SHELL is not bash-compatible; bash syntax would hang",
    ),
]


@pytest.fixture
def bash():
    """A live bash REPL, terminated even if a test fails."""
    language = Bash()
    language.start_process()
    try:
        yield language
    finally:
        language.terminate()


def _output(language, code):
    """Console output from a run, excluding OI's own state line.

    The state line is added by CwdTrackingMixin after every command, not by the
    command itself, and these tests are about what the *command's* output looks
    like once the marker has been consumed. Filtering it here keeps that
    distinction explicit instead of quietly loosening the assertions.
    """
    return [
        str(chunk["content"])
        for chunk in language.run(code)
        if chunk.get("format") != "active_line"
        and chunk.get("content")
        and not str(chunk["content"]).startswith("[Shell State:")
    ]


def test_output_without_trailing_newline_does_not_leak_the_marker(bash):
    """printf writes no trailing newline, so the marker lands mid-line.

    Before the fix this yielded 'no trailing newline##oi_cwd##/home/user/project'
    verbatim: the marker in the model's context and the absolute path on screen.
    """
    out = _output(bash, 'printf "no trailing newline"')

    assert out == ["no trailing newline"]
    assert not any("##oi_cwd##" in line for line in out)


def test_leaked_marker_never_shows_a_path(bash):
    """The path must be consumed, not merely the marker text.

    Checking only that the marker is gone would pass while the absolute path
    still leaked, which is the half of the problem that matters for privacy.
    """
    out = _output(bash, 'printf "no trailing newline"')

    assert not any(bash.cwd in line for line in out)


def test_cwd_still_tracks_after_a_newline_less_block(bash):
    """Consuming the marker must not stop it updating the tracked directory.

    The fix moved from an anchored match to a partition, so this guards against a
    future "just drop the line" shortcut that would silently break tracking.
    """
    _output(bash, 'printf "x"')

    assert bash.cwd


def test_normal_output_is_unchanged(bash):
    """The common case -- output ending in a newline -- must be byte-identical."""
    out = _output(bash, 'echo "ends with newline"')

    assert out == ["ends with newline\n"]


def test_cat_of_a_file_without_trailing_newline(bash, tmp_path):
    """The shape that actually surfaced it: cat of a file with no final newline.

    The reported case was reading a generated shell script, whose last line had
    no newline, so the marker was appended to it.
    """
    script = tmp_path / "script.sh"
    script.write_text("echo hi")  # no trailing newline

    out = _output(bash, f'cat "{script}"')

    assert not any("##oi_cwd##" in line for line in out)
    assert any(line.strip() == "echo hi" for line in out)


def test_marker_name_is_not_credential_shaped():
    """The marker must not read like a password tag.

    `##oi_pwd##` is a path probe, but `pwd` in the name plus the ##tag##
    delimiters made a leak of it indistinguishable from a leaked secret. Guard
    the name so it is not "tidied" back.
    """
    from interpreter.core.terminal.languages.cwd_tracking import _CWD_MARKER

    assert _CWD_MARKER == "##oi_cwd##"
    assert "pwd" not in _CWD_MARKER.lower()
    assert "pass" not in _CWD_MARKER.lower()


def test_marker_is_emitted_by_every_tracked_shell():
    """bash, cmd and PowerShell must all emit the marker the parser expects.

    They each hardcode the string, so renaming it can leave one shell emitting a
    marker nothing strips -- which reintroduces the leak for that shell only, and
    silently.

    Checked without instantiating: Cmd refuses to run off Windows, so
    constructing it would raise before the assertion. The echo is a pure string
    method, so read its source instead.
    """
    import inspect

    from interpreter.core.terminal.languages.bash import Bash as _Bash
    from interpreter.core.terminal.languages.cmd import Cmd as _Cmd
    from interpreter.core.terminal.languages.cwd_tracking import _CWD_MARKER
    from interpreter.core.terminal.languages.powershell import PowerShell as _PowerShell

    for cls in (_Bash, _Cmd, _PowerShell):
        source = inspect.getsource(cls._cwd_marker_echo)
        assert _CWD_MARKER in source, f"{cls.__name__} does not emit {_CWD_MARKER}"


class TestShellStateIsReportedToTheModel:
    """Shells must tell the model where they are, the way Python does.

    JupyterLanguage reports CWD, imported modules, variables and functions after
    every cell, and that report is the only way the model can tell that
    something it ran earlier had an effect. Shells reported nothing: the cwd was
    tracked in CwdTrackingMixin and then discarded, because the ##oi_cwd## marker
    is consumed precisely so the path never reaches the model's context. The
    model was left guessing after every `cd`.

    Cwd only. Shell variables and functions are rarely carried between turns and
    reading them would cost a round-trip per command; the cwd is already known.
    """

    def test_state_line_reports_the_current_cwd(self):
        """The line names the directory the shell is actually in."""
        bash = Bash()
        assert bash.cwd in bash._state_line()

    def test_state_line_follows_a_cd(self):
        """Tracking drives the report, so it cannot drift from the shell."""
        bash = Bash()
        bash.cwd = "/tmp/somewhere-else"
        assert "/tmp/somewhere-else" in bash._state_line()

    def test_state_line_is_emitted_after_a_real_command(self):
        """End to end through run(): the report reaches the consumer."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bash = Bash()
            try:
                chunks = list(bash.run(f"cd {tmp} && echo marker-ok"))
            finally:
                bash.terminate()
            joined = "\n".join(
                str(c.get("content", ""))
                for c in chunks
                if c.get("type") == "console"
            )
            assert "Shell State" in joined, f"no state line in {joined!r}"
            assert (
                tmp in joined
            ), "the reported cwd must be the directory we cd'd into"

    def test_state_line_does_not_reintroduce_the_marker(self):
        """The report carries the path, never the ##oi_cwd## marker.

        The marker exists to be consumed. Surfacing the state by echoing the
        marker line back out would undo the leak it was introduced to fix.
        """
        bash = Bash()
        assert "##oi_cwd##" not in bash._state_line()
