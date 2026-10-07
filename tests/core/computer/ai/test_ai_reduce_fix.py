"""Regression tests for query_reduce_chunks (#209).

The reducer looped while more than one response remained but never fed
the summaries back, so multi-response inputs looped forever — and a
single response fell through to an unbound summaries[0]. These pin the
fixed behavior: reduce to one, return it directly when already single.
"""

from types import SimpleNamespace
from unittest import mock

import interpreter.core.computer.ai.ai as ai_module
from interpreter.core.computer.ai.ai import query_reduce_chunks


def _stub_llm():
    return SimpleNamespace(model="gpt-4o-mini")


def test_reduce_multiple_responses_terminates():
    """Several responses reduce down to a single merged summary."""
    calls = []

    def fake_fast_llm(llm, query, chunk):
        calls.append(chunk)
        return f"summary-of({len(chunk)})"

    with mock.patch.object(ai_module, "fast_llm", side_effect=fake_fast_llm):
        result = query_reduce_chunks(
            ["alpha", "beta", "gamma", "delta"], _stub_llm(), 1000, "merge"
        )

    assert isinstance(result, str)
    assert result.startswith("summary-of(")
    assert calls, "expected at least one reduce pass"


def test_reduce_single_response_returns_it():
    """One response needs no reduction and is returned as-is."""
    with mock.patch.object(ai_module, "fast_llm") as fast_llm:
        result = query_reduce_chunks(["only"], _stub_llm(), 1000, "merge")

    assert result == "only"
    fast_llm.assert_not_called()
