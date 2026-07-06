from interpreter.core.computer.files.files import get_close_matches_in_text


def test_exact_phrase_ranked_first():
    """An exact phrase match is ranked first among close matches."""
    filedata = "The quick brown fox jumps over the lazy dog"
    matches = get_close_matches_in_text("quick brown fox", filedata)
    assert matches[0] == "quick brown fox"


def test_typo_returns_closest_phrases():
    """A typo query returns up to three closest phrases from the file text."""
    filedata = "foobar baz qux foobar"
    matches = get_close_matches_in_text("foobaz", filedata)
    assert len(matches) <= 3
    assert matches[0] == "foobar"


def test_respects_n_limit():
    """get_close_matches_in_text honors the n parameter for result count."""
    filedata = "one two three four five six seven eight"
    matches = get_close_matches_in_text("one two", filedata, n=1)
    assert len(matches) == 1


def test_empty_filedata_returns_empty():
    """Empty file text yields no close matches regardless of the query."""
    assert get_close_matches_in_text("anything", "") == []


def test_empty_original_text_matches_every_position():
    """An empty query matches every sliding window (including empty phrases) in file text."""
    matches = get_close_matches_in_text("", "one two three")
    assert matches == ["", "", ""]


def test_query_longer_than_filedata_returns_empty():
    """When the query has more words than the file, no phrase window can match."""
    assert get_close_matches_in_text("one two three four", "one two") == []


def test_n_zero_returns_empty():
    """n=0 requests zero results even when close matches exist."""
    assert get_close_matches_in_text("one", "one two three", n=0) == []
