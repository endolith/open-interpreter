from interpreter.terminal_interface.utils.find_image_path import find_image_path


def test_find_image_path_returns_all_existing(tmp_path):
    """Every existing image path mentioned in the text is returned."""
    short = tmp_path / "a.png"
    nested = tmp_path / "nested" / "longer_name.png"
    nested.parent.mkdir()
    short.write_bytes(b"x")
    nested.write_bytes(b"x")
    text = f"See {short} and {nested}"
    result = find_image_path(text)
    assert set(result) == {str(short), str(nested)}


def test_find_image_path_empty_list_when_missing():
    """find_image_path returns an empty list when no referenced image exists."""
    assert find_image_path("No images in this text.") == []


def test_find_image_path_empty_text_returns_empty_list():
    """find_image_path returns an empty list for empty input."""
    assert find_image_path("") == []


def test_find_image_path_matches_each_extension_case(tmp_path):
    """Path extensions are matched case-insensitively (png/jpg/jpeg and upper).

    The regex lists both cases explicitly; dropping the upper-case alternatives
    would miss real files saved with a capitalised extension.
    """
    for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"):
        path = tmp_path / f"image.{ext}"
        path.write_bytes(b"x")
        assert find_image_path(f"look at {path}") == [str(path)], ext


def test_find_image_path_returns_every_candidate(tmp_path):
    """Multiple existing candidates are all returned (no longest-only collapse)."""
    long_path = tmp_path / "aaaaaaaaaa.png"
    lexically_greater = tmp_path / "z.png"
    long_path.write_bytes(b"x")
    lexically_greater.write_bytes(b"x")

    result = find_image_path(f"{long_path} and {lexically_greater}")

    assert set(result) == {str(long_path), str(lexically_greater)}
