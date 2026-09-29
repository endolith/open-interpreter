import subprocess

import pytest

from interpreter.core.terminal.languages.bash import Bash
from interpreter.core.terminal.languages.resolve_bash import resolve_bash_executable


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
    assert run_preprocessed(code) == expected


@pytest.mark.parametrize("code", ["true", "true\n", "echo hi\n", "true\n# done\n"])
def test_success_is_reported_as_zero(code):
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
