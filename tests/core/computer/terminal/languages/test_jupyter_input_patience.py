"""Tests for the stalled-kernel input-patience mechanism in JupyterLanguage.

When a program produces no output for longer than `input_patience` seconds, the
iopub listener stops assuming the process is just quiet and asks a model whether
the user needs to type something into stdin — because a program blocked on
`input()` looks identical to one still computing. That question is a **paid LLM
call**, made from a background thread while the user waits.

Everything here is built on a stub kernel rather than a real one: the branch is
only reachable once a program has been quiet for 15+ seconds, and no test should
have to wait that long to check what it does.
"""

import queue
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.computer.terminal.languages import jupyter_language as jl
from interpreter.core.computer.terminal.languages.jupyter_language import JupyterLanguage


def _join_listener(lang, timeout=10):
    """Wait for the iopub listener, and never leave it running.

    The listener is a plain (non-daemon) thread, so a test that failed to stop it
    would not report a failure — it would hang the process at interpreter exit,
    turning an ordinary assertion error into a CI job that times out with no
    message. So the cleanup happens *before* the failure is raised, and only then
    is the failure raised.

    Raising is deliberately deferred to a second join: setting `finish_flag` is
    what lets the loop return, and the assertion records whether it got there on
    its own or only because of that.
    """
    lang.listener_thread.join(timeout=timeout)
    if lang.listener_thread.is_alive():
        lang.finish_flag = True
        lang.listener_thread.join(timeout=timeout)
    assert not lang.listener_thread.is_alive(), (
        "listener did not finish, even after being asked to stop"
    )


def _delta(content):
    """A litellm-shaped streaming chunk."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def _language(*, finish_after_first_poll=True, patience=15):
    """A JupyterLanguage with a stub kernel, built without starting a real one.

    `iopub_channel.get_msg` always raises `queue.Empty`, which is what an idle
    kernel looks like. On the first poll it optionally sets `finish_flag` so the
    listener returns instead of spinning forever.
    """
    lang = JupyterLanguage.__new__(JupyterLanguage)
    lang.computer = SimpleNamespace(
        interpreter=SimpleNamespace(
            stop_event=mock.Mock(is_set=mock.Mock(return_value=False)),
            messages=[],
            llm=SimpleNamespace(model="test-model", api_key="sk-test"),
        )
    )
    lang.km = SimpleNamespace(interrupt_kernel=mock.Mock())
    lang.kc = SimpleNamespace(
        execute=mock.Mock(),
        input=mock.Mock(),
        iopub_channel=SimpleNamespace(),
    )

    state = {"polls": 0}

    def get_msg(timeout=None):
        state["polls"] += 1
        if finish_after_first_poll:
            lang.finish_flag = True
        raise queue.Empty

    lang.kc.iopub_channel.get_msg = get_msg
    lang.poll_state = state
    # Pretend the program has been silent for an hour, so the patience branch
    # fires on the first listener iteration.
    lang.last_output_time = time.time() - 3600
    lang.last_output_message_time = time.time() - 3600
    lang.finish_flag = False
    lang._patience = patience
    return lang


def _run_listener(lang, monkeypatch, completion_response=None, api_key="sk-test"):
    """Run `_execute_code` and join the listener, returning the litellm mock."""
    lang.computer.interpreter.llm.api_key = api_key
    chunks = (
        [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=c))]) for c in completion_response]
        if completion_response is not None
        else []
    )
    completion = mock.Mock(return_value=chunks)
    monkeypatch.setattr(jl.litellm, "completion", completion)

    lang._execute_code("print(1)", queue.Queue())
    _join_listener(lang)
    return completion


def test_a_quiet_kernel_asks_the_model_whether_input_is_needed(monkeypatch):
    """Silence past the patience window triggers a question to the model.

    A program blocked on `input()` is indistinguishable from one still computing,
    so the listener asks. This is a real LLM call issued from the listener thread
    while the user is waiting, which is why the call's parameters are pinned
    rather than just its existence.
    """
    lang = _language()
    completion = _run_listener(lang, monkeypatch, completion_response=["no input needed"])

    assert completion.call_count == 1
    params = completion.call_args.kwargs
    assert params["stream"] is True
    assert params["temperature"] == 0, "this decision should be deterministic"
    assert params["model"] == "test-model"
    # The system message tells the model the expected output format.
    system = params["messages"][0]
    assert "<input></input>" in system["content"]


def test_input_from_the_model_is_typed_into_the_kernel(monkeypatch):
    """`<input>y</input>` in the response is typed into the waiting program."""
    lang = _language()
    _run_listener(lang, monkeypatch, completion_response=["<input>y</input>"])

    lang.kc.input.assert_called_once_with("y")
    assert lang.finish_flag is True, "the wait is over once input has been supplied"


def test_ctrl_c_stops_the_program_instead_of_typing(monkeypatch):
    """`<input>CTRL-C</input>` interrupts the kernel rather than typing text.

    Typing the literal string "CTRL-C" into a process would be useless — the
    point is to send an interrupt, so the marker must be handled as a signal and
    nothing must be typed.
    """
    lang = _language(finish_after_first_poll=False)
    _run_listener(lang, monkeypatch, completion_response=["<input>CTRL-C</input>"])

    assert lang.kc.input.call_count == 0, "CTRL-C must not be typed as text"
    # The stub is told not to set finish_flag itself, so the only thing that can
    # end the listener is the flag the CTRL-C branch sets — and ending it is what
    # triggers the interrupt. With the stub closing the loop, `finish_flag` and
    # the absence of a keystroke both held even if the response were ignored.
    lang.km.interrupt_kernel.assert_called_once()


def test_ctrl_c_is_matched_case_insensitively(monkeypatch):
    """The marker is uppercased before comparison, so `<input>ctrl-c</input>` works."""
    lang = _language(finish_after_first_poll=False)
    _run_listener(lang, monkeypatch, completion_response=["<input>ctrl-c</input>"])

    assert lang.kc.input.call_count == 0
    lang.km.interrupt_kernel.assert_called_once()


def test_a_response_without_an_input_tag_types_nothing(monkeypatch):
    """A reply with no `<input>` marker is ignored.

    The model is asked a yes/no-ish question and is told to use tags only when it
    means to type. Free-form prose must not be typed into the user's program.
    """
    lang = _language()
    _run_listener(lang, monkeypatch, completion_response=["I think it is still running."])

    assert lang.kc.input.call_count == 0
    assert lang.finish_flag is True


def test_non_string_stream_chunks_are_ignored(monkeypatch):
    """A `None` delta (what providers send as a role-only first chunk) is skipped.

    Providers open a stream with a chunk whose content is `None`. Accumulating it
    would raise `TypeError` on `response += content` and be swallowed by the
    retry handler, so the question would silently never be asked.
    """
    lang = _language()
    monkeypatch.setattr(
        jl.litellm,
        "completion",
        mock.Mock(
            return_value=[
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
                # A tagged chunk *after* the None: if accumulation raised on the
                # None, the listener would catch it and retry, and the old
                # "nothing was typed" assertion would still pass. Reaching the
                # second chunk is what proves the None was skipped.
                _delta("<input>y</input>"),
            ]
        ),
    )
    lang._execute_code("print(1)", queue.Queue())
    _join_listener(lang)

    lang.kc.input.assert_called_once_with("y")


def test_the_api_key_is_only_sent_when_one_is_configured(monkeypatch):
    """No `api_key` parameter is passed when the interpreter has no key.

    Passing `api_key=None` is not the same as omitting it for every provider, so
    the parameter's presence is pinned rather than its value.
    """
    lang = _language()
    completion = _run_listener(lang, monkeypatch, completion_response=[], api_key=None)

    assert "api_key" not in completion.call_args.kwargs


def test_the_api_key_is_forwarded_when_configured(monkeypatch):
    """A configured key is passed through to the model call."""
    lang = _language()
    completion = _run_listener(lang, monkeypatch, completion_response=[])

    assert completion.call_args.kwargs["api_key"] == "sk-test"


def test_the_frozen_hint_is_added_only_for_a_long_stall(monkeypatch):
    """Past 500 seconds the prompt suggests CTRL-C; before that it does not.

    The distinction is the whole point of the second sentence: for a brief pause
    the model should type input, but after eight minutes of silence the prompt
    should raise the possibility that the process is simply stuck.
    """
    lang = _language()
    lang.last_output_time = time.time() - 600
    lang.last_output_message_time = time.time() - 600
    completion = _run_listener(lang, monkeypatch, completion_response=[])

    prompt = completion.call_args.kwargs["messages"][1]["content"]
    assert "CTRL-C" in prompt
    assert "frozen" in prompt


def test_a_brief_silence_gets_no_frozen_hint(monkeypatch):
    """A stall under the 500s mark asks the question without the frozen hint."""
    lang = _language()
    lang.last_output_time = time.time() - 100
    lang.last_output_message_time = time.time() - 100
    completion = _run_listener(lang, monkeypatch, completion_response=[])

    prompt = completion.call_args.kwargs["messages"][1]["content"]
    assert "frozen" not in prompt
    assert "running for over 15 seconds" in prompt


def test_the_patience_window_is_configurable(monkeypatch):
    """`INTERPRETER_TERMINAL_INPUT_PATIENCE` sets the window, and it is honoured.

    A window that did not widen would make the mechanism fire on any momentary
    pause, which is what makes the default of 15 seconds load-bearing.
    """
    monkeypatch.setenv("INTERPRETER_TERMINAL_INPUT_PATIENCE", "3600")
    lang = _language()
    # Silently for 100s: inside the default 15s window would have fired, but
    # comfortably inside a 3600s one.
    lang.last_output_time = time.time() - 100
    lang.last_output_message_time = time.time() - 100
    completion = _run_listener(lang, monkeypatch, completion_response=[])

    assert completion.call_count == 0, "should not have asked inside a 3600s window"


def test_a_recent_message_suppresses_the_question(monkeypatch):
    """Both clocks must be stale before asking.

    `last_output_message_time` is the throttle: it is reset whenever the question
    is asked, so a long silence asks once rather than on every listener iteration.
    """
    lang = _language()
    lang.last_output_time = time.time() - 3600
    lang.last_output_message_time = time.time()  # just asked
    completion = _run_listener(lang, monkeypatch, completion_response=[])

    assert completion.call_count == 0, "the question must not repeat immediately"
