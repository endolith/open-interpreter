import io
import subprocess
import threading
import time

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


def test_block_after_redirecting_shell_stdout_does_not_hang():
    """`exec >/dev/null` closes the stdout pipe while the shell stays alive.

    Every later end marker goes to /dev/null and the stdout reader has already
    exited at EOF, so a next block written to that same live shell would never
    report done and run() would spin forever. The live shell must be replaced
    instead, and its turn must not claim the shell exited.

    Driven from a watchdog thread so a regression fails instead of hanging the
    suite.
    """
    language = Bash()
    turns = []
    finished = threading.Event()

    def _drive():
        try:
            turns.append(_text_contents(list(language.run("exec >/dev/null\n"))))
            turns.append(_text_contents(list(language.run("echo revived\n"))))
        finally:
            finished.set()

    threading.Thread(target=_drive, daemon=True).start()
    try:
        assert finished.wait(timeout=20), f"run() hung; turns so far: {turns!r}"
    finally:
        language.terminate()
    first, second = turns
    assert not any("the shell exited" in text for text in first)
    assert any("closed its output" in text for text in first)
    assert any("revived" in text for text in second)


class _ShellCaughtMidExit:
    """A shell that has closed its pipes but cannot be reaped yet.

    A real exit closes the process's files before it becomes reapable, so
    `poll()` at stdout EOF usually still says "running" (measured: 259 of 300
    `exit 7` runs). This fake reproduces that order deterministically: running
    for `delay` seconds, then exited with `returncode`.
    """

    def __init__(self, stdout, returncode, delay=0.2):
        self.stdout = stdout
        self.stdin = io.StringIO()
        self.returncode = returncode
        self._reapable_at = time.monotonic() + delay
        self.terminated = False

    def poll(self):
        if time.monotonic() >= self._reapable_at:
            return self.returncode
        return None

    def wait(self, timeout=None):
        remaining = self._reapable_at - time.monotonic()
        if timeout is not None and remaining > timeout:
            time.sleep(timeout)
            raise subprocess.TimeoutExpired("bash", timeout)
        time.sleep(max(0.0, remaining))
        return self.returncode

    def terminate(self):
        self.terminated = True


def test_exiting_shell_seen_running_at_eof_reports_its_code():
    """A shell still being torn down at EOF is reported as exited, not killed.

    Reading poll() once at EOF mistook these for live shells, losing the exit
    code (the `exit 7` tests above were flaky because of it) and, worse, would
    route a dying shell into the live-shell-with-closed-stdout branch.
    """
    language = Bash()
    stream = io.StringIO("##active_line1##\n")
    language.process = _ShellCaughtMidExit(stream, returncode=7)
    process = language.process

    language.handle_stream_output(stream, False)

    texts = []
    while not language.output_queue.empty():
        texts.append(str(language.output_queue.get_nowait().get("content")))
    assert language.done.is_set()
    assert any("the shell exited with code 7" in text for text in texts), texts
    assert not process.terminated


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
