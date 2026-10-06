"""Commands that wait for a human must fail fast instead of hanging the turn.

Two mechanisms, because the two languages differ in what they can do.

**POSIX shells** get their command's stdin pointed at the null device --
`{ cat; } < /dev/null` -- so anything that reads stdin sees EOF immediately.
Verified: a bare `cat` blocks forever without it and returns in 0.01s with it,
and `read -p "x: " v` likewise. The redirect is scoped to the command body so the
shell's own stdin, the channel commands arrive on, is untouched: the next command
still runs. Set per-shell (`command_stdin_from_null_device`) because it needs
shell syntax -- cmd.exe cannot parse braces, and PowerShell needs its own form.

**Python** gets an AST check instead. input() in a kernel cell blocks on a read
nothing will ever satisfy: the cell never returns, the end-of-execution marker is
never emitted, and the turn hangs with the reader thread alive and waiting, so
Ctrl-C cannot break it either. Confirmed still hanging after 40s with the 15s
stdin side-channel disabled. The null-device trick cannot be used here because
the kernel deliberately has a mechanism for supplying input.

Rejection is preferred over support for both: an agent asking a question belongs
in its reply text, and there may be nobody at the keyboard to answer.
"""

import os
import shutil

import pytest

from interpreter.core.terminal.languages.bash import Bash
from interpreter.core.terminal.languages.jupyter_language import (
    _BLOCKING_INPUT_MESSAGE,
    _blocking_input_call,
)
from interpreter.core.terminal.languages.subprocess_language import SubprocessLanguage

_BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(
    not _BASH, reason="bash not installed"
)


# --------------------------------------------------------------------------
# Shell: stdin from the null device
# --------------------------------------------------------------------------


def test_only_posix_shells_opt_in():
    """The redirect needs shell syntax, so it must be opt-in per shell.

    cmd.exe cannot parse `{ ...; }`, and PowerShell needs a different form, so
    leaving this on by default in the base class would break both.
    """
    assert Bash.command_stdin_from_null_device is True
    assert SubprocessLanguage.command_stdin_from_null_device is False


def test_wrapper_is_syntactically_valid_for_single_and_multi_line():
    """Both shapes must produce a complete bash group.

    The terminator differs by whether the code already ends in a newline. A
    newline followed by `;` is a syntax error ("near unexpected token `;'"), and
    multi-line blocks do end in one; `{ cmd }` without the `;` is equally an
    error. Getting this wrong breaks every command, not just interactive ones.
    """
    bash = Bash()
    assert bash._wrap_command_for_null_stdin("echo hi") == "{ echo hi; } < /dev/null"
    assert (
        bash._wrap_command_for_null_stdin("echo one\necho two\n")
        == "{ echo one\necho two\n} < /dev/null"
    )


def _run_bash(code, timeout=15):
    """Run `code` in a live bash, failing if it does not finish in time.

    A deadline is not optional here. The regression these tests guard against
    manifests as a *hang*, not an error: a malformed wrapper leaves bash waiting
    on a command that never completes, so a test that simply calls run() blocks
    forever instead of failing. That is the same failure mode the feature exists
    to prevent, reproduced inside the suite that verifies it.

    The worker thread cannot be killed if it does block, so it is left to die
    with the process; the point is that the test reports rather than hangs.
    """
    import threading

    language = Bash()
    chunks = []
    finished = threading.Event()

    def _go():
        try:
            chunks.extend(list(language.run(code)))
        except Exception as exc:  # noqa: BLE001 - surfaced in the assertion below
            chunks.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=_go, daemon=True)
    worker.start()
    completed = finished.wait(timeout=timeout)
    language.terminate()
    assert completed, f"bash did not finish within {timeout}s (hung): {code!r}"
    texts = [
        str(c.get("content", ""))
        for c in chunks
        if isinstance(c, dict) and c.get("format") != "active_line" and c.get("content")
    ]
    return "\n".join(texts)


@needs_bash
@pytest.mark.parametrize(
    "code",
    [
        'echo hello',
        'echo abcdef | grep -o cd',          # pipe supplies its own stdin
        "awk 'BEGIN { print 1 }'",           # braces in a quoted program
        'find /etc -maxdepth 0 -name hostname',
        'echo one\necho two',                # multi-line
        'echo $(echo nested)',              # command substitution
        'X=5; echo "X=$X"',                 # semicolon-joined
        'echo "{"',                         # unbalanced brace, quoted
    ],
)
def test_real_commands_still_work_wrapped(code):
    """Wrapping must not break ordinary commands.

    A regression here would be severe and silent: every bash command the agent
    runs would fail. The brace cases matter because the wrapper adds its own.
    """
    joined = _run_bash(code)
    assert "syntax error" not in joined.lower(), f"{code!r} broke: {joined!r}"
    assert "unexpected token" not in joined.lower(), f"{code!r} broke: {joined!r}"


@needs_bash
def test_command_channel_survives_the_redirect():
    """The next command must still arrive after a redirected one.

    This is the whole reason the redirect is scoped to the command body rather
    than applied to the subprocess stdin, which is how commands are delivered.
    Redirecting at Popen would end the session after one command.
    """
    assert _run_bash('cat') is not None       # would block without the redirect
    assert "CHANNEL_ALIVE" in _run_bash('echo CHANNEL_ALIVE')


@needs_bash
def test_interactive_command_returns_instead_of_hanging():
    """`read` completes at EOF rather than waiting for a keystroke that never comes."""
    joined = _run_bash('read -p "name: " v && echo "GOT $v"')
    # It returns; it just does not print GOT, because read exits non-zero at EOF
    # and the && short-circuits. That is the documented trade.
    assert "##end_of_execution##" not in joined
    assert "GOT " not in joined


# --------------------------------------------------------------------------
# Python: reject input() before the kernel wedges
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ('x = input("a: ")', "input"),
        ("y = raw_input()", "raw_input"),
        ("z = builtins.input()", "builtins.input"),
        ("def g():\n    return input('x')", "input"),   # nested still counts
        ("input = 5\nprint(input)", None),             # a variable, not a call
        ("# do not use input() here\nprint(1)", None),  # a comment
        ("f = obj.input(1)", None),                    # some other object's
        ('print("fine")', None),
        ("def f(:", None),                              # unparseable: pass through
    ],
)
def test_detector_distinguishes_calls_from_lookalikes(source, expected):
    """Only real calls to a stdin-reading builtin are caught.

    Text matching would flag `input = 5` and a comment mentioning input(), both
    of which are ordinary code the agent should be allowed to write.
    """
    assert _blocking_input_call(source) == expected


def test_message_tells_the_model_what_to_do_instead():
    """The refusal has to be actionable, or it just stalls the loop differently."""
    message = _BLOCKING_INPUT_MESSAGE.format(name="input")
    assert "ask the user" in message.lower()
    assert "stdin" in message.lower()


@needs_bash
def test_flag_off_means_code_is_written_verbatim():
    """A language with the flag off must get its code through untouched.

    Checked at the dispatch point rather than on the wrapper helper, which wraps
    unconditionally -- write_block_to_stdin is what consults the flag. Getting
    this wrong would redirect languages that cannot parse the syntax at all.
    """

    class _Stdin:
        def __init__(self):
            self.written = []

        def write(self, payload):
            self.written.append(payload)

        def flush(self):
            pass

    class _Off(SubprocessLanguage):
        pass

    lang = _Off()
    lang.process = type("P", (), {"stdin": _Stdin()})()
    lang.write_block_to_stdin("echo hi")

    assert lang.process.stdin.written == ["echo hi\n"], lang.process.stdin.written


def test_null_device_path_matches_the_platform():
    """The redirect target must exist on the platform this runs on."""
    bash = Bash()
    assert os.path.exists(bash.null_device_path), bash.null_device_path

@needs_bash
def test_run_rejects_input_before_the_kernel_is_touched():
    """The guard must be wired into run(), not merely present as a helper.

    Found by mutation: stubbing the detector's result out entirely passed every
    other test in this file, because they all exercised
    _blocking_input_call directly. The detector could be dead code and the suite
    would still be green -- which is the precise failure the feature exists to
    prevent.
    """
    import threading

    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    interpreter.terminal.terminate()
    collected = []
    finished = threading.Event()

    def _go():
        try:
            for chunk in interpreter.terminal.run(
                "python", 'name = input("project name: ")\nprint(name)',
                display=False,
            ):
                collected.append(str(chunk.get("content", "")))
        finally:
            finished.set()

    worker = threading.Thread(target=_go, daemon=True)
    worker.start()
    try:
        assert finished.wait(timeout=90), "the input() cell never returned"
    finally:
        interpreter.terminal.terminate()

    joined = "\n".join(collected)
    assert "stdin" in joined, f"expected the refusal, got: {joined[:200]!r}"
    assert "Ask the user" in joined, joined[:200]


@needs_bash
def test_kernel_still_usable_after_a_rejected_cell():
    """A refused cell must not leave the kernel wedged.

    The rejection happens before execution, so nothing should have been sent --
    but "nothing was sent" and "the kernel is still healthy" are different claims
    and only the second one matters to the next turn.
    """
    import threading

    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    interpreter.terminal.terminate()

    def _collect(code):
        out = []
        finished = threading.Event()

        def _go():
            try:
                for chunk in interpreter.terminal.run(
                    "python", code, display=False
                ):
                    out.append(str(chunk.get("content", "")))
            finally:
                finished.set()

        threading.Thread(target=_go, daemon=True).start()
        assert finished.wait(timeout=90), f"cell never finished: {code!r}"
        return "\n".join(out)

    try:
        _collect('name = input("x: ")')
        after = _collect('print("KERNEL_STILL_ALIVE")')
    finally:
        interpreter.terminal.terminate()

    assert "KERNEL_STILL_ALIVE" in after, after[:200]
