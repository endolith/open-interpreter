"""A hung or runaway shell command must be killed, completely, and reported once.

SubprocessLanguage.run() waits for an end-of-execution marker that a hung
command never prints. The idle timeout (INTERPRETER_COMMAND_IDLE_TIMEOUT) and
the optional total cap (INTERPRETER_COMMAND_TIMEOUT) end that wait by killing
the shell and everything it started.
"""

import subprocess
from unittest import mock

import pytest

from interpreter.core.terminal.languages import subprocess_language
from interpreter.core.terminal.languages.subprocess_language import SubprocessLanguage


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
