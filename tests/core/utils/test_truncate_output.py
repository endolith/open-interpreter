from interpreter.core.utils.truncate_output import truncate_output


def test_short_output_unchanged():
    data = "hello world"
    assert truncate_output(data, max_output_chars=2800) == data


def test_long_output_truncated_with_marker():
    data = "a" * 5000
    result = truncate_output(data, max_output_chars=2800)
    assert result.startswith("Output truncated (5,000 characters total)")
    assert "[...]" in result
    assert result.count("a") < len(data)


def test_retruncate_reapplies_truncation():
    data = "x" * 5000
    first = truncate_output(data, max_output_chars=100)
    second = truncate_output(first, max_output_chars=100)
    assert "Output truncated" in second
    assert "[...]" in second


def test_add_scrollbars_appends_hint():
    data = "b" * 5000
    result = truncate_output(data, max_output_chars=100, add_scrollbars=True)
    assert "get_last_output()" in result


def test_exactly_at_limit_not_truncated():
    data = "c" * 2800
    assert truncate_output(data, max_output_chars=2800) == data
