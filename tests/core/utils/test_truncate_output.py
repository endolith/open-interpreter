from interpreter.core.utils.truncate_output import truncate_output


def test_short_output_unchanged():
    """Output within the character limit is returned verbatim."""
    data = "hello world"
    assert truncate_output(data, max_output_chars=2800) == data


def test_long_output_truncated_with_marker():
    """Long output is replaced with head + [...] + tail and a banner."""
    data = "a" * 5000
    result = truncate_output(data, max_output_chars=2800)
    assert result.startswith("Output truncated (5,000 characters total)")
    assert "[...]" in result
    assert result.count("a") < len(data)


def test_retruncate_keeps_truncated_shape():
    """A second truncate pass on shortened output should still look truncated.

    truncate_output only strips a prior banner when it matches the newly built
    banner text (which embeds len(data)). After the first pass the character
    count changes, so a second banner may appear — we assert the result still
    has the middle ellipsis and is much shorter than the original input.
    """
    data = "x" * 5000
    first = truncate_output(data, max_output_chars=100)
    second = truncate_output(first, max_output_chars=100)
    assert "[...]" in second
    assert len(second) < len(data)
    assert "Output truncated" in second


def test_add_scrollbars_appends_hint():
    """add_scrollbars=True appends a get_last_output() hint to the truncation banner."""
    data = "b" * 5000
    result = truncate_output(data, max_output_chars=100, add_scrollbars=True)
    assert "get_last_output()" in result


def test_exactly_at_limit_not_truncated():
    """Output at exactly the character limit is not truncated."""
    data = "c" * 2800
    assert truncate_output(data, max_output_chars=2800) == data


def test_unicode_output_unchanged_when_short():
    """Emoji and other non-ASCII characters survive unchanged in short output."""
    assert truncate_output("Done ✅") == "Done ✅"


def test_unicode_output_truncates_without_error():
    """Long emoji output truncates without error; head/tail still contain valid emoji."""
    data = "✅" * 2000
    result = truncate_output(data, max_output_chars=100)
    assert result.startswith("Output truncated")
    assert "✅" in result
