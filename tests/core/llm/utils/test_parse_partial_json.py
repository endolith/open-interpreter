from interpreter.core.llm.utils.parse_partial_json import parse_partial_json

import pytest


def test_parse_complete_json():
    """Valid complete JSON objects parse to the expected dict."""
    assert parse_partial_json('{"language": "python", "code": "print(1)"}') == {
        "language": "python",
        "code": "print(1)",
    }


def test_parse_empty_string_returns_none():
    """An empty input string cannot be parsed and returns None."""
    assert parse_partial_json("") is None


def test_parse_truncated_object_closes_brace():
    """A truncated object missing its closing brace is repaired and parsed successfully."""
    result = parse_partial_json('{"language": "python", "code": "print(1)')
    assert result == {"language": "python", "code": "print(1)"}


def test_parse_truncated_string_closes_quote():
    """A truncated string value missing its closing quote is repaired and parsed successfully."""
    result = parse_partial_json('{"language": "python", "code": "hello')
    assert result == {"language": "python", "code": "hello"}


def test_parse_unclosed_string_with_newline():
    """Newlines inside an unclosed string value are preserved when the string is auto-closed."""
    result = parse_partial_json('{"code": "line1\nline2')
    assert result == {"code": "line1\nline2"}


def test_parse_mismatched_brace_returns_none():
    """Extra closing braces that cannot be repaired cause parse_partial_json to return None."""
    assert parse_partial_json('{"a": 1}}') is None


def test_parse_unrecoverable_garbage_returns_none():
    """Input that is not JSON-like at all returns None instead of raising."""
    assert parse_partial_json("not json at all {{{") is None


def test_parse_truncated_array():
    """A truncated array missing its closing bracket is repaired and parsed successfully."""
    result = parse_partial_json("[1, 2, 3")
    assert result == [1, 2, 3]


def test_parse_none_raises_type_error():
    """None is a caller bug, not incomplete JSON — must raise, not return None."""
    with pytest.raises(TypeError):
        parse_partial_json(None)
