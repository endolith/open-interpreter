"""Tests for ai.query_reduce_chunks, including its #209 defects."""

import multiprocessing
from types import SimpleNamespace

import pytest

from interpreter.core.computer.ai.ai import query_reduce_chunks


def _stub_llm():
    """An llm stand-in whose fast_llm path always summarizes instantly."""
    interpreter = SimpleNamespace(
        messages=[],
        system_message="system",
        chat=_instant_summary,
    )
    return SimpleNamespace(model="gpt-4o-mini", interpreter=interpreter)


def _instant_summary(message):
    """A chat() replacement that summarizes instantly (module-level: picklable)."""
    return [{"content": "summary"}]


def _reduce_worker(responses, chunk_size, query):
    """Run the reducer in a child process (module-level: picklable target)."""
    query_reduce_chunks(responses, _stub_llm(), chunk_size, query)


def test_query_reduce_chunks_single_response_raises_name_error():
    """One response skips the loop, so the unbound `summaries` raises NameError.

    query_reduce_chunks only assigns `summaries` inside
    `while len(responses) > 1`; a single response falls through to
    `return summaries[0]` with nothing bound. Part of the #209
    characterization.
    """
    with pytest.raises(NameError):
        query_reduce_chunks(["only"], _stub_llm(), 100, "q")


@pytest.mark.linux_ci
@pytest.mark.timeout(60)
def test_query_reduce_chunks_never_terminates_on_multiple():
    """Multiple responses loop forever: summaries are recomputed, never fed back.

    Each pass re-chunks the ORIGINAL responses and discards the summaries
    instead of reducing them, so len(responses) never shrinks. Runs in a
    child process with instant stubs — still alive after the grace period
    proves non-termination, and terminate() leaves no spinning thread behind
    to pollute teardown. Part of the #209 characterization.
    """
    worker = multiprocessing.Process(
        target=_reduce_worker, args=(["first", "second"], 100, "q")
    )
    worker.start()
    try:
        worker.join(timeout=10)
        assert worker.is_alive()
    finally:
        worker.terminate()
        worker.join()
