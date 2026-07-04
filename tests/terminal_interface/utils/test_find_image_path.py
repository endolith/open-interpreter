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
