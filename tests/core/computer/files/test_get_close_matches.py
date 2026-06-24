from interpreter.core.computer.files.files import get_close_matches_in_text


def test_exact_phrase_ranked_first():
    filedata = "The quick brown fox jumps over the lazy dog"
    matches = get_close_matches_in_text("quick brown fox", filedata)
    assert matches[0] == "quick brown fox"


def test_typo_returns_closest_phrases():
    filedata = "foobar baz qux foobar"
    matches = get_close_matches_in_text("foobaz", filedata)
    assert len(matches) <= 3
    assert any("foobar" in match for match in matches)


def test_respects_n_limit():
    filedata = "one two three four five six seven eight"
    matches = get_close_matches_in_text("one two", filedata, n=1)
    assert len(matches) == 1


def test_empty_filedata_returns_empty():
    assert get_close_matches_in_text("anything", "") == []
