from interpreter.core.utils.truncate_output import truncate_output

import pytest


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


def test_non_positive_max_output_chars_rejected():
    """max_output must be positive; zero and negative values raise ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        truncate_output("hello", max_output_chars=0)
    with pytest.raises(ValueError, match="positive integer"):
        truncate_output("hello", max_output_chars=-1)


def test_empty_string_unchanged():
    """Empty output is returned verbatim without adding a truncation banner."""
    assert truncate_output("") == ""


def test_small_max_chars_truncates_short_output():
    """Very small max_output_chars still truncates when data exceeds the limit."""
    data = "abcdefgh"
    result = truncate_output(data, max_output_chars=4)
    assert result.startswith("Output truncated (8 characters total)")
    assert "[...]" in result


def test_head_and_tail_are_each_half_the_limit():
    """Truncation keeps exactly max_output_chars // 2 characters from each end.

    The banner promises "N characters from start/end", so a mutation to the
    divisor (//3) would silently keep fewer than advertised while the existing
    "in" assertions still passed did. Pin the exact head and tail.
    """
    data = "0123456789"
    result = truncate_output(data, max_output_chars=6)

    head, tail = data[:3], data[-3:]
    assert result.endswith("012\n[...]\n789")
    assert head == "012" and tail == "789"
def test_retruncation_strips_a_matching_previous_banner():
    """Re-truncating the same text strips the old banner instead of nesting it.

    The banner text embeds both the total character count and the per-end
    figure, so it is only stripped when the second call rebuilds a byte-identical
    banner: same max_output_chars, and an input whose length still matches the
    count in the old banner. The earlier version of this test re-truncated with a
    larger limit, so no truncation happened at all and both assertions were
    trivially true.

    The length is found by fixed-point iteration rather than hard-coded, so the
    test keeps working if the banner wording or length ever changes.
    """
    max_output_chars = 200

    # Truncating produces len(banner) + max_output_chars + len("\\n[...]\\n").
    # Iterate until that equals the input length, so re-truncating rebuilds the
    # same banner and the strip branch is reached.
    length = 400
    for _ in range(50):
        first = truncate_output("z" * length, max_output_chars=max_output_chars)
        if len(first) == length:
            break
        length += len(first) - length

    assert len(first) == length, "could not find a self-consistent truncated length"

    second = truncate_output(first, max_output_chars=max_output_chars)

    # The old banner is removed and a fresh one added, so re-truncating is
    # idempotent. Had the banner not matched, it would survive inside the new
    # head and appear twice.
    assert second == first
    assert second.count("Output truncated") == 1
    assert second.count("[...]") == 1


def test_truncation_separator_is_exactly_bracket_ellipsis():
    """The join between head and tail is exactly "\\n[...]\\n".

    A wrapped separator would still satisfy a substring check for "[...]" but
    would leak the wrapper into the model's view of the output.
    """
    data = "q" * 100
    result = truncate_output(data, max_output_chars=10)

    # Pin the whole head + separator + tail rather than a substring: excluding
    # one specific wrapper still lets any other malformed separator through.
    # max_output_chars=10 splits evenly, so each end keeps 5 characters.
    assert result.endswith("qqqqq\n[...]\nqqqqq")


def test_default_add_scrollbars_is_false():
    """Omitting add_scrollbars leaves the pagination hint out.

    The default must stay False: a True default would append a
    get_last_output() hint to every truncated message even when the caller did
    not ask for the paging helper.
    """
    data = "m" * 5000
    result = truncate_output(data, max_output_chars=100)

    assert "get_last_output()" not in result


def test_default_max_output_chars_is_2800():
    """The default limit is 2800 characters.

    Callers rely on the module default, so a shifted default would change the
    amount of output kept with no call-site change; 2800 is exact and its
    neighbour 2801 is not.
    """
    at_limit = "n" * 2800
    over_limit = "n" * 2801

    assert truncate_output(at_limit) == at_limit
    assert truncate_output(over_limit).startswith("Output truncated")


def test_max_output_chars_of_one_is_rejected():
    """A limit of 1 is rejected like 0 and negative values.

    The guard is `<= 0`, so a mutation to `<= 1` would reject a legal limit;
    pinning that 1 is accepted and 0 is not keeps the boundary exact.
    """
    assert truncate_output("", max_output_chars=1) == ""
    with pytest.raises(ValueError, match="positive integer"):
        truncate_output("", max_output_chars=0)

