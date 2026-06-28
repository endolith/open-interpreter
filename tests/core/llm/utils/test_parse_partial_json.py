import json

import pytest

from interpreter.core.llm.utils.parse_partial_json import parse_partial_json


def test_parse_complete_json():
    assert parse_partial_json('{"language": "python", "code": "print(1)"}') == {
        "language": "python",
        "code": "print(1)",
    }


def test_parse_empty_string_returns_none():
    assert parse_partial_json("") is None


def test_parse_truncated_object_closes_brace():
    result = parse_partial_json('{"language": "python", "code": "print(1)')
    assert result == {"language": "python", "code": "print(1)"}


def test_parse_truncated_string_closes_quote():
    result = parse_partial_json('{"language": "python", "code": "hello')
    assert result == {"language": "python", "code": "hello"}


def test_parse_unclosed_string_with_newline():
    result = parse_partial_json('{"code": "line1\nline2')
    assert result == {"code": "line1\nline2"}


def test_parse_mismatched_brace_returns_none():
    assert parse_partial_json('{"a": 1}}') is None


def test_parse_unrecoverable_garbage_returns_none():
    assert parse_partial_json("not json at all {{{") is None


def test_parse_truncated_array():
    result = parse_partial_json("[1, 2, 3")
    assert result == [1, 2, 3]
