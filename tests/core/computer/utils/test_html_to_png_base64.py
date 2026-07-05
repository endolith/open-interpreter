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
