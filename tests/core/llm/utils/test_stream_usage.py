from types import SimpleNamespace

from interpreter.core.llm.utils.stream_usage import (
    _to_plain,
    _usage_details_for_display,
    _usage_from_chunk,
    format_last_usage_markdown,
    record_stream_chunk_usage,
)


def test_record_stream_chunk_usage_from_mapping():
    """A usage mapping on the final streamed chunk is stored on the llm."""
    llm = SimpleNamespace(last_completion_usage=None)
    record_stream_chunk_usage(
        llm,
        {"usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}},
    )

    assert llm.last_completion_usage == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }


def test_record_stream_chunk_usage_from_object():
    """A LiteLLM usage object is normalized before it is stored."""
    class Usage:
        def model_dump(self):
            return {"prompt_tokens": 2, "total_tokens": 2}

    llm = SimpleNamespace(last_completion_usage=None)
    record_stream_chunk_usage(llm, SimpleNamespace(usage=Usage()))

    assert llm.last_completion_usage == {"prompt_tokens": 2, "total_tokens": 2}


def test_record_stream_chunk_usage_without_usage_is_a_noop():
    """Chunks without usage must not clear the last recorded usage."""
    llm = SimpleNamespace(last_completion_usage={"total_tokens": 1})
    record_stream_chunk_usage(llm, {"choices": []})
    record_stream_chunk_usage(llm, {"usage": None})

    assert llm.last_completion_usage == {"total_tokens": 1}


def test_format_last_usage_markdown_includes_breakdowns():
    """Usage output distinguishes reported zeroes from unreported detail values."""
    text = format_last_usage_markdown(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "audio_tokens": None,
            },
            "completion_tokens_details": {"reasoning_tokens": 5},
            "provider_field": "kept",
        }
    )

    assert "Prompt tokens:** 100" in text
    assert "Completion tokens:** 20" in text
    assert "Prompt — Cached Tokens:** 0" in text
    assert "Prompt — Audio Tokens:** *not reported*" in text
    assert "Completion — Reasoning Tokens:** 5" in text
    assert '"provider_field": "kept"' in text


def test_format_last_usage_markdown_falls_back_to_raw_usage():
    """Unknown-only usage still renders instead of producing an empty panel."""
    text = format_last_usage_markdown({"custom": {"nested": None}})

    assert '"custom"' in text
    assert '"nested": "(not reported)"' in text


def test_to_plain_handles_lists_and_nested_values():
    """Lists and tuples are normalized recursively for usage details."""
    assert _to_plain([1, {"nested": (2, None)}]) == [1, {"nested": [2, None]}]


def test_to_plain_falls_back_when_all_conversion_methods_fail():
    """A non-serializable object still yields a printable string."""

    class Broken:
        def model_dump(self):
            raise ValueError("no model_dump")

        def dict(self):
            raise ValueError("no dict")

        def model_dump_json(self):
            raise ValueError("no json")

        def __str__(self):
            return "fallback-value"

    assert _to_plain(Broken()) == "fallback-value"


def test_usage_from_chunk_returns_none_for_missing_or_empty_usage():
    """Chunks without a usage object do not create a false usage record."""
    assert _usage_from_chunk(None) is None
    assert _usage_from_chunk({"usage": {}}) is None


def test_usage_details_for_display_normalizes_lists():
    """Nested lists in provider-specific fields get the same readable null treatment."""
    assert _usage_details_for_display([None, {"x": None}]) == [
        "(not reported)",
        {"x": "(not reported)"},
    ]


def test_format_last_usage_markdown_empty_usage_uses_json_fallback():
    """An empty/unrecognized usage object still renders its raw payload."""
    text = format_last_usage_markdown({})

    assert "```json" in text
    assert "{}" in text
