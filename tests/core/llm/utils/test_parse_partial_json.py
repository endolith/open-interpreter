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


def test_parse_whitespace_only_returns_none():
    """Whitespace-only input is not valid JSON and returns None."""
    assert parse_partial_json("   ") is None


def test_parse_truncated_nested_object():
    """A truncated nested object is repaired by closing inner and outer braces."""
    result = parse_partial_json('{"a": {"b": 1')
    assert result == {"a": {"b": 1}}


def test_parse_truncated_after_closed_array():
    """A closed array followed by a truncated string repairs correctly, requiring bracket matching in the repair path."""
    result = parse_partial_json('{"a": [1], "b": "x')
    assert result == {"a": [1], "b": "x"}


def test_parse_truncated_unclosed_array():
    """A truncated array value is auto-closed in the right order."""
    result = parse_partial_json('{"a": [1, 2')
    assert result == {"a": [1, 2]}


def test_parse_mismatched_closing_bracket_returns_none():
    """A closing bracket that does not match the innermost open structure is malformed and returns None."""
    assert parse_partial_json('{"a": ]}') is None


def test_parse_escaped_quote_inside_truncated_string():
    """An escaped quote inside a truncated string value is kept as content, not treated as a string terminator."""
    result = parse_partial_json('{"code": "print(\\"hello')
    assert result == {"code": 'print("hello'}


def test_parse_escaped_backslash_inside_truncated_string():
    """An escaped backslash inside a truncated string value is preserved as a single backslash."""
    result = parse_partial_json('{"path": "C:\\\\temp')
    assert result == {"path": "C:\\temp"}


def test_parse_empty_string_value():
    """An empty string value parses correctly, including in the repair path."""
    assert parse_partial_json('{"a": ""}') == {"a": ""}
    result = parse_partial_json('{"a": ""')
    assert result == {"a": ""}


def test_parse_escape_completed_at_truncation_boundary():
    """A backslash pair that completes exactly at the truncation boundary survives repair.

    The two trailing backslashes form a complete escape, so the repair closes
    the string cleanly and keeps one backslash as content.
    """
    result = parse_partial_json('{"a": "ends with backslash \\\\')
    assert result == {"a": "ends with backslash \\"}


def test_parse_dangling_backslash_at_truncation_boundary():
    """A single trailing backslash is a genuinely unterminated escape and yields None.

    The repair cannot close the string without inventing a second backslash, so
    json.loads still fails and the parser reports failure instead of guessing.
    This is a real limitation of the repair path, not a bug in the input.
    """
    assert parse_partial_json('{"a": "ends with backslash \\') is None


def test_parse_truncated_after_empty_string_value():
    """An empty string value followed by more truncated content parses correctly (empty-string quote pairing)."""
    result = parse_partial_json('{"a": "", "b": 1')
    assert result == {"a": "", "b": 1}


def test_parse_truncated_after_closed_nested_object():
    """A closed nested object followed by a truncated string repairs correctly (inner brace popped)."""
    result = parse_partial_json('{"a": {}, "b": "x')
    assert result == {"a": {}, "b": "x"}
