from unittest import mock

from interpreter.core.computer.utils import html_to_png_base64


def test_html_to_png_base64_returns_base64(tmp_path, monkeypatch):
    """html_to_png_base64 renders HTML to PNG and returns base64-encoded bytes."""
    png_bytes = b"\x89PNG\r\n\x1a\nfake"
    monkeypatch.setattr(
        html_to_png_base64, "get_storage_path", lambda: str(tmp_path)
    )

    mock_hti = mock.Mock()
    mock_hti.output_path = None

    def fake_screenshot(html_str, save_as, size):
        (tmp_path / save_as).write_bytes(png_bytes)

    mock_hti.screenshot = fake_screenshot

    with mock.patch.object(html_to_png_base64, "html2image") as lazy:
        lazy.Html2Image.return_value = mock_hti
        result = html_to_png_base64.html_to_png_base64("<html></html>")

    import base64

    assert result == base64.b64encode(png_bytes).decode()
    assert not list(tmp_path.glob("*.png"))


def test_html_to_png_base64_screenshots_at_the_documented_size(tmp_path, monkeypatch):
    """The screenshot is requested at 960x540 and written to the storage path.

    The size is the fixed capture geometry; a changed or dropped size would
    render the page at html2image's default viewport instead.
    """
    monkeypatch.setattr(
        html_to_png_base64, "get_storage_path", lambda: str(tmp_path)
    )
    seen = {}

    mock_hti = mock.Mock()
    mock_hti.output_path = None

    def fake_screenshot(html_str, save_as, size):
        seen["html_str"] = html_str
        seen["size"] = size
        (tmp_path / save_as).write_bytes(b"x")

    mock_hti.screenshot = fake_screenshot

    with mock.patch.object(html_to_png_base64, "html2image") as lazy:
        lazy.Html2Image.return_value = mock_hti
        html_to_png_base64.html_to_png_base64("<p>hi</p>")

    assert seen["html_str"] == "<p>hi</p>"
    assert seen["size"] == (960, 540)
    assert mock_hti.output_path == str(tmp_path)


def test_html_to_png_base64_temp_filename_is_ten_digits_and_png(tmp_path, monkeypatch):
    """The temp file is ten random digits with a lowercase ".png" suffix.

    The name is what os.path.join and open() later look up, so a changed length
    or suffix (k=11, ".PNG", wrapped digits) would leave the file unfound or
    mis-typed even though the bytes were written elsewhere.
    """
    monkeypatch.setattr(
        html_to_png_base64, "get_storage_path", lambda: str(tmp_path)
    )
    seen = {}

    mock_hti = mock.Mock()
    mock_hti.output_path = None

    def fake_screenshot(html_str, save_as, size):
        seen["save_as"] = save_as
        (tmp_path / save_as).write_bytes(b"x")

    mock_hti.screenshot = fake_screenshot

    with mock.patch.object(html_to_png_base64, "html2image") as lazy:
        lazy.Html2Image.return_value = mock_hti
        html_to_png_base64.html_to_png_base64("<html></html>")

    name = seen["save_as"]
    assert name.endswith(".png")
    assert not name.endswith(".PNG")
    stem = name[:-4]
    assert len(stem) == 10
    assert stem.isdigit()

