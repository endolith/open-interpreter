from interpreter.terminal_interface.utils.find_image_path import find_image_path


def test_find_image_path_returns_longest_existing(tmp_path):
    """When multiple image paths appear in text, the longest existing path wins."""
    short = tmp_path / "a.png"
    nested = tmp_path / "nested" / "longer_name.png"
    nested.parent.mkdir()
    short.write_bytes(b"x")
    nested.write_bytes(b"x")
    text = f"See {short} and {nested}"
    result = find_image_path(text)
    assert result == str(nested)


def test_find_image_path_none_when_missing():
    """find_image_path returns None when no referenced image file exists."""
    assert find_image_path("No images in this text.") is None


def test_find_image_path_empty_text_returns_none():
    """find_image_path returns None for empty input."""
    assert find_image_path("") is None


def test_find_image_path_matches_each_extension_case(tmp_path):
    """Path extensions are matched case-insensitively (png/jpg/jpeg and upper).

    The regex lists both cases explicitly; dropping the upper-case alternatives
    would miss real files saved with a capitalised extension.
    """
    for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"):
        path = tmp_path / f"image.{ext}"
        path.write_bytes(b"x")
        assert find_image_path(f"look at {path}") == str(path), ext


def test_find_image_path_prefers_the_longest_existing_path(tmp_path):
    """Among several existing candidates, the longest string is returned.

    The selection is max(..., key=len). The two names below are chosen so the
    lexical and length orders disagree ("z.png" sorts above "aaaaaaaaaa.png"
    but is shorter): a dropped key would pick the lexically greatest, not the
    longest.
    """
    long_path = tmp_path / "aaaaaaaaaa.png"
    lexically_greater = tmp_path / "z.png"
    long_path.write_bytes(b"x")
    lexically_greater.write_bytes(b"x")

    result = find_image_path(f"{long_path} and {lexically_greater}")

    assert result == str(long_path)
    assert len(str(long_path)) > len(str(lexically_greater))
    assert str(long_path) < str(lexically_greater)

