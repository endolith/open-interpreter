"""A hung or runaway shell command must be killed, completely, and reported once.

SubprocessLanguage.run() waits for an end-of-execution marker that a hung
command never prints. The idle timeout (INTERPRETER_COMMAND_IDLE_TIMEOUT) and
the optional total cap (INTERPRETER_COMMAND_TIMEOUT) end that wait by killing
the shell and everything it started.
"""

import shutil
import subprocess
import threading
from unittest import mock

import pytest

from interpreter.core.terminal.languages import subprocess_language
from interpreter.core.terminal.languages.bash import Bash
from interpreter.core.terminal.languages.subprocess_language import SubprocessLanguage

needs_bash = pytest.mark.skipif(not shutil.which("bash"), reason="bash not installed")


def _run(language, code, deadline=15):
    """Run `code` to completion and return its output text, failing on a hang.

    The bugs pinned here show up as run() never returning, so the run happens
    on a worker thread with a deadline: a regression is a failed assertion,
    not a stuck suite. The shell is left running so a test can run a second
    command in it; callers terminate it.
    """
    chunks = []
    finished = threading.Event()

    def _go():
        try:
            chunks.extend(language.run(code))
        finally:
            finished.set()

    threading.Thread(target=_go, daemon=True).start()
    completed = finished.wait(timeout=deadline)
    if not completed:
        language.terminate()
    assert completed, f"run() did not return within {deadline}s: {code!r}"
    return "".join(
        str(c.get("content"))
        for c in chunks
        if c.get("format") != "active_line" and c.get("content")
    )


# --------------------------------------------------------------------------
# Windows: the kill must reach the shell's descendants
# --------------------------------------------------------------------------


def _language_with_live_process(pid=4242):
    """A SubprocessLanguage whose process looks alive, without spawning one."""
    language = SubprocessLanguage()
    language.process = mock.Mock(pid=pid)
    language.process.poll.return_value = None
    return language


def test_windows_kill_takes_the_whole_process_tree():
    """On Windows the kill goes through `taskkill /T /F`, not proc.kill().

    proc.kill() is TerminateProcess on the shell alone; CREATE_NEW_PROCESS_GROUP
    does not change that. A hung ssh or installer the shell started would
    survive the timeout while the message claimed it was terminated.
    Windows cannot be exercised here, so this pins the dispatch with mocks.
    """
    language = _language_with_live_process(pid=4242)
    with mock.patch.object(subprocess_language.os, "name", "nt"), mock.patch.object(
        subprocess_language.subprocess, "run"
    ) as run:
        language._kill_process_group()

    run.assert_called_once()
    assert run.call_args.args[0] == ["taskkill", "/T", "/F", "/PID", "4242"]
    language.process.kill.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("taskkill"),
        subprocess.TimeoutExpired("taskkill", 10),
        subprocess.CalledProcessError(1, "taskkill"),
    ],
    ids=["missing", "hung", "nonzero-exit"],
)
def test_windows_kill_falls_back_to_the_shell_when_taskkill_fails(failure):
    """If taskkill is missing, hangs or fails, the shell itself is still killed.

    Losing the descendants is bad; leaving the shell alive as well would leave
    run() with a live process it has already reported as killed.
    """
    language = _language_with_live_process()
    with mock.patch.object(subprocess_language.os, "name", "nt"), mock.patch.object(
        subprocess_language.subprocess, "run", side_effect=failure
    ):
        language._kill_process_group()

    language.process.kill.assert_called_once()


@pytest.mark.parametrize(
    "os_name, says, never_says",
    [("posix", "SIGKILL", "taskkill"), ("nt", "taskkill", "SIGKILL")],
)
def test_timeout_message_names_the_kill_this_platform_uses(os_name, says, never_says):
    """The timeout message describes the kill that actually ran.

    It used to say "SIGKILL to the process group" everywhere, which is false on
    Windows: there are no signals or process groups to kill there.
    """
    with mock.patch.object(subprocess_language.os, "name", os_name):
        description = subprocess_language._tree_kill_description()

    assert says in description
    assert never_says not in description


# --------------------------------------------------------------------------
# The total cap must fire even while the command keeps printing
# --------------------------------------------------------------------------


@needs_bash
def test_total_timeout_kills_a_command_that_never_stops_printing(monkeypatch):
    """INTERPRETER_COMMAND_TIMEOUT stops a runaway that prints continuously.

    The idle timeout cannot catch `yes`, a runaway loop or `tail -f` on a busy
    log, because they never go quiet; the total cap exists for exactly those.
    It used to be checked only when the output queue came up empty, which a
    command printing at least every 0.3s never allows, so run() never returned.
    """
    monkeypatch.setenv("INTERPRETER_COMMAND_TIMEOUT", "2")
    language = Bash()
    try:
        output = _run(language, "while true; do echo tick; sleep 0.05; done")
    finally:
        language.terminate()

    assert "tick" in output
    assert "Command killed: it exceeded the 2s total time limit" in output


# --------------------------------------------------------------------------
# A killed command's output stays with that command, and its shell is reaped
# --------------------------------------------------------------------------


@needs_bash
def test_output_read_after_the_kill_is_not_reported_as_the_next_commands(monkeypatch):
    """What a killed command printed belongs to its own result, not the next.

    A command hung on a prompt has usually printed the prompt without a
    newline, so the reader cannot hand it over until the kill closes the pipe.
    The queue used to be drained immediately after the kill, before the reader
    got that EOF; the prompt -- the one clue to why it hung -- then sat in the
    queue and came out as the start of the next command's output.
    """
    monkeypatch.setenv("INTERPRETER_COMMAND_IDLE_TIMEOUT", "1")
    language = Bash()
    try:
        first = _run(language, "printf 'Continue? [y/N] '; sleep 30")
        second = _run(language, "echo second")
    finally:
        language.terminate()

    assert "Continue? [y/N]" in first
    assert "Continue?" not in second, f"killed command's output leaked: {second!r}"
    assert "second" in second


@needs_bash
def test_killed_shell_is_reaped_and_its_pipes_closed(monkeypatch):
    """A timeout kill waits for the shell and closes its pipes.

    Dropping the Popen reference instead left a zombie until some later Popen
    happened to reap it, and left the stdin pipe open until garbage collection.
    """
    monkeypatch.setenv("INTERPRETER_COMMAND_IDLE_TIMEOUT", "1")
    language = Bash()
    language.start_process()
    process = language.process
    try:
        _run(language, "sleep 30")
    finally:
        language.terminate()

    assert process.returncode is not None, "killed shell was never reaped"
    assert process.stdin.closed
