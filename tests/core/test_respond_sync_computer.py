"""Tests for the computer state sync that surrounds code execution in respond().

When ``sync_computer`` is on, respond() pushes the interpreter's computer state
into the code being run before it executes, and reads it back afterwards, so
that user code sees the same desktop, notes and apps the interpreter has. Both
directions are wrapped in a bare ``except Exception`` that prints and continues,
which means a sync failure must never be able to abort a turn.
"""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.respond import respond

SYNC_IN_MARKER = "computer.load_dict(json.loads("
SYNC_OUT_MARKER = "computer_dict = computer.to_dict()"


class FakeLanguage:
    def __init__(self, name="Python", file_extension="py"):
        self.name = name
        self.file_extension = file_extension


def _sync_interpreter(
    *,
    language="python",
    sync_computer=True,
    debug=False,
    to_dict_value=None,
    run_result=None,
    run_error=None,
):
    """An interpreter with sync_computer on and a recording `computer.run`.

    `run` is told apart by its arguments: the two sync calls pass plain strings
    and no `stream` kwarg, while user code is run with `stream=True`. That lets
    one recorder cover all three.
    """
    calls = []
    state = {"sync_out_attempted": False}
    to_dict_value = {} if to_dict_value is None else to_dict_value

    def run(lang, code_to_run, **run_kwargs):
        calls.append({"lang": lang, "code": code_to_run, **run_kwargs})
        if SYNC_OUT_MARKER in code_to_run:
            state["sync_out_attempted"] = True
        if run_error is not None and run_kwargs.get("stream"):
            raise run_error
        if SYNC_OUT_MARKER in code_to_run:
            return run_result if run_result is not None else []
        return iter([{"type": "console", "format": "output", "content": "42"}])

    loaded = []

    computer = SimpleNamespace(
        terminal=SimpleNamespace(
            languages=[FakeLanguage()],
            # Any language resolves, so a test can drive execution with a
            # non-python language and reach the `language == "python"` gate
            # rather than being rejected as unsupported first.
            get_language=lambda lang: FakeLanguage(lang.capitalize(), "txt"),
        ),
        import_computer_api=False,
        system_message="",
        run=run,
        load_dict=loaded.append,
        verbose=False,
        debug=False,
        emit_images=False,
        max_output=2800,
        save_skills=True,
        to_dict=lambda: to_dict_value,
    )

    messages = [
        {"role": "assistant", "type": "code", "format": language, "content": "1+1"}
    ]

    interpreter = SimpleNamespace(
        system_message="You are helpful.",
        custom_instructions="",
        messages=messages,
        computer=computer,
        llm=SimpleNamespace(run=lambda msgs: iter([]),
                            supports_vision=False),
        verbose=False,
        debug=debug,
        auto_run=True,
        loop=False,
        loop_message="continue",
        loop_breakers=[],
        sync_computer=sync_computer,
        offline=True,
        os=False,
        display_message=mock.Mock(),
        max_budget=0,
        max_output=2800,
    )
    def _end_the_turn(_msgs):
        """Yield a plain-text reply and append it, so respond() terminates.

        respond() re-reads the last message after code runs. If the model adds
        nothing, the same code message is still last and the code executes again,
        forever. Appending to interpreter.messages (rather than a captured list,
        which respond() replaces) is what ends the turn.
        """
        reply = {"type": "message", "content": "done"}
        interpreter.messages.append(
            {"role": "assistant", "type": "message", "content": "done"}
        )
        yield reply

    interpreter.loaded = loaded
    interpreter.calls = calls
    interpreter.state = state
    interpreter.llm.run = _end_the_turn
    return interpreter


def _run_to_sync_out(interpreter, limit=40):
    """Consume respond() until the read-back has been attempted, then return.

    respond() is a `while True` generator: after running code it goes back to the
    top and re-reads the last message, which is still the code message, so it
    would run the same code again indefinitely. Nothing in this module terminates
    that loop — the caller in production drives it with a fresh LLM turn. So these
    tests advance the generator only as far as the sync they care about and stop,
    which is also what the existing tests in test_respond.py do with islice.

    Returns the messages yielded up to that point.
    """
    messages = []
    generator = respond(interpreter)
    for _ in range(limit):
        if interpreter.state["sync_out_attempted"]:
            break
        try:
            messages.append(next(generator))
        except StopIteration:
            break
    # Deliberately not closed. respond() wraps its execution block in a bare
    # `except:` that yields a traceback, so it swallows GeneratorExit and tries to
    # yield again, which Python rejects with "generator ignored GeneratorExit".
    # Abandoning the generator, as the existing tests in test_respond.py do with
    # islice, avoids provoking that path.
    return messages


def _drain(interpreter):
    """Run to completion of a single code execution, bounded."""
    return _run_to_sync_out(interpreter)


def test_state_is_pushed_into_the_code_before_it_runs():
    """The interpreter's computer dict is serialised into the run, ahead of user code.

    Order matters: the sync has to be its own `run` call, and it has to precede
    the user's code, or the code would see an empty computer.
    """
    interpreter = _sync_interpreter(to_dict_value={"os": "linux", "apps": {}})
    _drain(interpreter)

    user_code = [c for c in interpreter.calls if c.get("stream")]
    assert len(user_code) == 1
    sync_calls = [c for c in interpreter.calls if SYNC_IN_MARKER in c["code"]]
    assert len(sync_calls) == 1, "state should be pushed exactly once"
    assert interpreter.calls.index(sync_calls[0]) < interpreter.calls.index(user_code[0])
    assert '"os": \\"linux\\"' in sync_calls[0]["code"] or '"os": "linux"' in sync_calls[0]["code"]


def test_hashes_and_system_message_are_stripped_before_syncing():
    """`_hashes` and `system_message` are dropped, so user code gets a clean load.

    Both are interpreter bookkeeping rather than user-visible state, and
    `load_dict` would otherwise receive them.
    """
    interpreter = _sync_interpreter(
        to_dict_value={"os": "linux", "_hashes": {"abc": 1}, "system_message": "hi"}
    )
    _drain(interpreter)

    sync_code = [c for c in interpreter.calls if SYNC_IN_MARKER in c["code"]][0]["code"]
    assert "_hashes" not in sync_code
    assert "system_message" not in sync_code
    assert "linux" in sync_code


def _explode_on_sync_out(interpreter, error):
    """Replace `computer.run` so the read-back raises, and flag the attempt."""
    original_run = interpreter.computer.run

    def run(lang, code_to_run, **run_kwargs):
        if SYNC_OUT_MARKER in code_to_run:
            interpreter.state["sync_out_attempted"] = True
            raise error
        return original_run(lang, code_to_run, **run_kwargs)

    interpreter.computer.run = run
    return run


def test_a_sync_failure_in_debug_mode_is_yielded_as_a_traceback_not_raised(capsys):
    """In debug mode the re-raise is intercepted anyway, and a traceback is yielded.

    The sync block reads `except Exception as e: if interpreter.debug: raise`,
    which looks like it should escape `debug=True`. It does not: the whole
    execution block is wrapped in a bare `except:` that catches the re-raised
    error and yields it as an output message. So `interpreter.debug` does not
    change what the caller observes here — it changes whether the same error also
    prints a warning first.

    Pinned as observed, because the code reads as though it should raise and
    someone will otherwise "fix" the test to match the reading.
    """
    interpreter = _sync_interpreter(debug=True)
    _explode_on_sync_out(interpreter, RuntimeError("computer went away"))

    messages = _drain(interpreter)

    tracebacks = [m for m in messages if m.get("type") == "console" and "Traceback" in str(m.get("content"))]
    assert tracebacks, f"expected a yielded traceback, got {[str(m)[:60] for m in messages]}"
    assert "computer went away" in tracebacks[-1]["content"]


def test_a_sync_failure_in_normal_mode_prints_a_warning_and_continues(capsys):
    """Without debug, the read-back failure is reported and the turn carries on.

    The user's code has already run and its output already yielded by this point,
    so the bare `except` must not turn a working turn into a failed one. This is
    the behaviour the whole `except` block exists for.
    """
    interpreter = _sync_interpreter(debug=False)
    _explode_on_sync_out(interpreter, RuntimeError("computer went away"))

    messages = _drain(interpreter)

    assert any(m.get("content") == "42" for m in messages), "user output must survive"
    captured = capsys.readouterr().out
    assert "computer went away" in captured
    assert "Failed to sync your Computer" in captured
    assert not any("Traceback" in str(m.get("content")) for m in messages)


def test_no_sync_happens_when_sync_computer_is_off():
    """With the flag off, neither direction runs.

    Off is the default, so this is the path most turns take; it must not pay for
    two extra interpreter invocations.
    """
    interpreter = _sync_interpreter(sync_computer=False)
    _drain(interpreter)

    assert not any(SYNC_IN_MARKER in c["code"] for c in interpreter.calls)
    assert not any(SYNC_OUT_MARKER in c["code"] for c in interpreter.calls)
    assert len(interpreter.loaded) == 0


def test_state_is_not_synced_for_non_python_code():
    """Both directions are gated on `language == "python"`, and the gate matters.

    Only Python's interpreter has a `computer` module to load state into; syncing
    for another language would push `computer.load_dict(...)` into, say, a shell
    command. The fake terminal resolves any language here so execution actually
    proceeds and reaches the gate — resolving only "python" would make respond
    reject the language as unsupported and never get near it.
    """
    interpreter = _sync_interpreter(language="javascript")
    _drain(interpreter)

    assert not any(SYNC_IN_MARKER in c["code"] for c in interpreter.calls), (
        "state must not be pushed for a non-python language"
    )
    assert not any(SYNC_OUT_MARKER in c["code"] for c in interpreter.calls), (
        "state must not be read back for a non-python language"
    )
    # The code itself still ran — this is a gate on syncing, not on execution.
    assert any(c.get("stream") for c in interpreter.calls)


def test_state_is_synced_for_python_code():
    """The complement of the gate above: python does sync, in both directions."""
    interpreter = _sync_interpreter(language="python")
    _drain(interpreter)

    assert any(SYNC_IN_MARKER in c["code"] for c in interpreter.calls)
    assert any(SYNC_OUT_MARKER in c["code"] for c in interpreter.calls)


def test_state_is_read_back_after_the_code_runs():
    """The read-back lands on load_dict after user code, so user changes persist.

    Without this the interpreter's computer would be silently reset on the next
    turn, discarding anything the code just did to the desktop.
    """
    interpreter = _sync_interpreter(run_result=[{"content": '{"os": "darwin"}'}])
    _drain(interpreter)

    user_code = [c for c in interpreter.calls if c.get("stream")]
    sync_out = [c for c in interpreter.calls if SYNC_OUT_MARKER in c["code"]]
    assert len(sync_out) == 1
    assert interpreter.calls.index(sync_out[0]) > interpreter.calls.index(user_code[0])
    assert interpreter.loaded == [{"os": "darwin"}]
