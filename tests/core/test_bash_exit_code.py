import subprocess

import pytest

from interpreter.core.terminal.languages.bash import Bash
from interpreter.core.terminal.languages.resolve_bash import resolve_bash_executable
from interpreter.core.terminal.languages.shell_preprocess import preprocess_shell


@pytest.fixture(autouse=True)
def _require_bash():
    """Skip this module where no GNU bash exists (e.g. bare Windows CI).

    Every helper here boots a real shell; without one Bash() raises
    FileNotFoundError, which would fail the suite instead of skipping it.
    """
    try:
        resolve_bash_executable()
    except FileNotFoundError:
        pytest.skip("bash is not installed")


def run_preprocessed(code):
    """Preprocess `code` the way Bash does, run it, and return the reported status.

    Runs the generated script in a real shell rather than asserting on its text:
    the whole point is what `$?` holds by the time the end marker echoes it, and
    only bash can answer that.
    """
    script = Bash().preprocess_code(code)
    completed = subprocess.run(
        [resolve_bash_executable()],
        input=script,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return Bash().detect_exit_code(completed.stdout)


@pytest.mark.parametrize(
    "code",
    [
        "false",
        # Trailing newline. add_active_line_prints splits on "\n", so this used
        # to leave a marker echo as the last command before $? was read, and the
        # failure was reported as 0.
        "false\n",
        "false\n\n\n",
        # A trailing comment runs nothing either.
        "false\n# all done\n",
        "false  # inline comment",
    ],
)
def test_failure_is_reported_however_the_block_ends(code):
    """A failing last command reports 1 no matter what follows it.

    Blank lines and comments run nothing, but their marker echoes used to sit
    after the last real command and reset `$?` to 0 before it was captured.
    """
    assert run_preprocessed(code) == 1


@pytest.mark.parametrize(
    "code,expected",
    [
        ("(exit 7)\n", 7),
        ("bash -c 'exit 42'\n", 42),
        ("ls /definitely/not/a/real/path\n", 2),
        # The status of the whole chain, not of its first command.
        ("false || true\n", 0),
        ("true && false\n", 1),
    ],
)
def test_specific_exit_codes_survive(code, expected):
    """Statuses other than 0/1 pass through the marker uncapped and unclamped."""
    assert run_preprocessed(code) == expected


@pytest.mark.parametrize("code", ["true", "true\n", "echo hi\n", "true\n# done\n"])
def test_success_is_reported_as_zero(code):
    """Successful blocks report 0, including with trailing blank/comment lines."""
    assert run_preprocessed(code) == 0


def test_multiline_constructs_report_their_status():
    # has_multiline_commands() disables active-line injection here, so this
    # exercises the path without any injected echoes at all.
    assert run_preprocessed("if true; then false; fi\n") == 1
    assert run_preprocessed("for i in 1 2; do true; done\n") == 0


def test_blank_and_comment_lines_get_no_active_line_marker():
    out = Bash().preprocess_code("echo one\n\n# a comment\necho two\n")
    assert '##active_line1##' in out  # echo one
    assert '##active_line2##' not in out  # blank
    assert '##active_line3##' not in out  # comment
    assert '##active_line4##' in out  # echo two


def test_active_line_numbers_still_match_the_source():
    """Skipped lines keep their index so highlighting points at the right line."""
    out = Bash().preprocess_code("\n\necho third")
    assert '##active_line3##' in out


def test_detect_exit_code_parses_the_marker():
    bash = Bash()
    assert bash.detect_exit_code("##end_of_execution##0") == 0
    assert bash.detect_exit_code("##end_of_execution##130") == 130
    assert bash.detect_exit_code("trailing text ##end_of_execution##2") == 2
    assert bash.detect_exit_code("##end_of_execution##") is None
    assert bash.detect_exit_code("no marker here") is None


def test_heredoc_containing_marker_text_is_preserved():
    """Only the appended end marker is rewritten; user content is untouched.

    The user's code may itself contain the same echo line (e.g. inside a
    heredoc). Rewriting every occurrence would corrupt their output; only the
    final, pipeline-appended marker may carry the exit status.
    """
    out = Bash().preprocess_code('cat <<EOF\necho "##end_of_execution##"\nEOF\n')
    assert out.count("##end_of_execution##$__oi_exit_code") == 1
    assert run_preprocessed('cat <<EOF\necho "##end_of_execution##"\nEOF\n') == 0


def _run_block(code):
    """Run `code` through a persistent Bash session and return every chunk.

    Unlike run_preprocessed(), this drives the real session machinery: it
    proves the turn terminates (done gets set) even when the shell dies
    mid-block before emitting the end marker.
    """
    language = Bash()
    try:
        return list(language.run(code))
    finally:
        language.terminate()


def _text_contents(chunks):
    return [
        chunk.get("content", "")
        for chunk in chunks
        if isinstance(chunk.get("content"), str)
    ]


def test_shell_exit_terminates_the_turn_and_reports_status():
    """`exit 7` kills the persistent shell before the end marker runs.

    The turn must still end, and report the shell's status, instead of
    spinning on the queue forever. The next command restarts the session.
    """
    texts = _text_contents(_run_block("exit 7\n"))
    assert any("the shell exited with code 7" in text for text in texts)


def test_errexit_before_marker_terminates_the_turn():
    """`set -e` plus a failure exits the shell before the capture commands.

    Same requirement as an explicit exit: the turn ends and the status is
    reported rather than hanging.
    """
    texts = _text_contents(_run_block("set -e\nfalse\n"))
    assert any("the shell exited with code 1" in text for text in texts)


def test_session_restarts_after_shell_death():
    """A block after a shell-killing one runs in a fresh session.

    The dead process cannot take another write, so run() restarts it via its
    write retry; the follow-up block completes normally.
    """
    language = Bash()
    try:
        list(language.run("exit 7\n"))
        texts = _text_contents(list(language.run("echo revived\n")))
    finally:
        language.terminate()
    assert any("revived" in text for text in texts)


def test_cmd_comment_styles_get_no_marker_but_hash_lines_do():
    """Comment detection follows the shell: `#` is bash-only.

    Cmd uses `rem`/`::`, so a `#` line there is an ordinary (failing) command
    that keeps its highlight, while `rem` and `::` lines are skipped.
    """
    hashed = preprocess_shell(
        "# not a cmd comment\necho hi\n", comment_prefixes=("rem", "::")
    )
    assert "##active_line1##" in hashed
    assert "##active_line2##" in hashed
    remarked = preprocess_shell(
        "rem done\necho hi\n", comment_prefixes=("rem", "::")
    )
    assert "##active_line1##" not in remarked
    assert "##active_line2##" in remarked
    labeled = preprocess_shell(
        ":: label\necho hi\n", comment_prefixes=("rem", "::")
    )
    assert "##active_line1##" not in labeled
    assert "##active_line2##" in labeled
