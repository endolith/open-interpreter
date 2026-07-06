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
