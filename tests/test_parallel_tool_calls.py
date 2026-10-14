"""Regression tests for parallel tool calls sharing one response (#415).

The tool-call pipeline executes a single call per turn. When a response
carries several tool_calls entries, the first runs immediately and the
rest wait on the interpreter, served one per later turn without spending
further LLM calls — instead of being silently dropped.
"""

import json
from types import SimpleNamespace

from interpreter.core.llm.run_tool_calling_llm import run_tool_calling_llm


def _tool_entry(call_id, code, index=None):
    function = SimpleNamespace(
        name="execute",
        arguments=json.dumps({"language": "python", "code": code}),
    )
    if index is None:
        return SimpleNamespace(id=call_id, function=function)
    return SimpleNamespace(id=call_id, index=index, function=function)


def _fake_llm(chunks):
    interpreter = SimpleNamespace(
        verbose=False,
        computer=SimpleNamespace(terminal=SimpleNamespace(languages=[])),
    )
    return SimpleNamespace(
        interpreter=interpreter,
        completions=lambda **kwargs: iter(chunks),
    )


def _code_contents(chunks):
    return "".join(
        chunk["content"] for chunk in chunks if chunk["type"] == "code"
    )


def test_parallel_tool_calls_first_runs_and_rest_queues():
    """The first entry of a two-call response runs; the second is queued."""
    first = _tool_entry("call_1", "print('one')")
    second = _tool_entry("call_2", "print('two')")
    llm = _fake_llm([{"choices": [{"delta": {"tool_calls": [first, second]}}]}])

    chunks = list(
        run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "go"}]})
    )

    assert "print('one')" in _code_contents(chunks)
    assert "print('two')" not in _code_contents(chunks)
    queued = llm.interpreter._pending_tool_calls
    assert len(queued) == 1
    assert queued[0]["id"] == "call_2"
    assert "print('two')" in queued[0]["arguments"]


def test_queued_tool_call_runs_without_new_llm_call():
    """A queued call is served next turn even if the LLM is unreachable."""
    llm = _fake_llm([{"choices": [{"delta": {"tool_calls": [
        _tool_entry("call_1", "print('one')"),
        _tool_entry("call_2", "print('two')"),
    ]}}]}])
    list(run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "go"}]}))

    def _unreachable(**kwargs):
        raise AssertionError("no LLM call should be made for queued work")

    llm.completions = _unreachable
    chunks = list(
        run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "go"}]})
    )

    assert "print('two')" in _code_contents(chunks)
    assert getattr(llm.interpreter, "_pending_tool_calls", []) == []


def test_single_tool_call_leaves_no_queue():
    """Ordinary single-call responses behave exactly as before."""
    llm = _fake_llm([{"choices": [{"delta": {"tool_calls": [
        _tool_entry("call_1", "print('one')"),
    ]}}]}])

    chunks = list(
        run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "go"}]})
    )

    assert "print('one')" in _code_contents(chunks)
    assert getattr(llm.interpreter, "_pending_tool_calls", []) == []


def test_streamed_entries_in_separate_chunks_queue_by_id():
    """One entry per chunk (the streaming shape) still queues extras.

    The first id seen becomes the primary call; later ids wait, even when
    each arrives in its own chunk with no shared delta.
    """
    first = _tool_entry("call_1", "print('one')", index=0)
    second = _tool_entry("call_2", "print('two')", index=1)
    llm = _fake_llm([
        {"choices": [{"delta": {"tool_calls": [first]}}]},
        {"choices": [{"delta": {"tool_calls": [second]}}]},
    ])

    chunks = list(
        run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "go"}]})
    )

    assert "print('one')" in _code_contents(chunks)
    assert "print('two')" not in _code_contents(chunks)
    queued = llm.interpreter._pending_tool_calls
    assert len(queued) == 1
    assert "print('two')" in queued[0]["arguments"]


def test_streamed_fragments_concatenate_before_queueing():
    """A queued call split across chunks arrives whole.

    Continuation chunks carry no id, only the index: fragments sharing it
    concatenate into one queued call instead of queuing twice.
    """
    full = json.dumps({"language": "python", "code": "print('two')"})
    head = SimpleNamespace(
        id="call_2",
        index=1,
        function=SimpleNamespace(name="execute", arguments=full[:20]),
    )
    tail = SimpleNamespace(
        index=1,
        function=SimpleNamespace(name="execute", arguments=full[20:]),
    )
    llm = _fake_llm([
        {"choices": [{"delta": {"tool_calls": [
            _tool_entry("call_1", "print('one')", index=0)
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [head]}}]},
        {"choices": [{"delta": {"tool_calls": [tail]}}]},
    ])

    list(run_tool_calling_llm(llm, {"messages": [{"role": "user", "content": "go"}]}))

    queued = llm.interpreter._pending_tool_calls
    assert len(queued) == 1
    assert json.loads(queued[0]["arguments"])["code"] == "print('two')"
