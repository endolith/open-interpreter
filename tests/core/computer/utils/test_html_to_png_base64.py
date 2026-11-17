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


def test_html_to_png_base64_asks_the_browser_to_stay_quiet(tmp_path, monkeypatch):
    """The browser wrapper is constructed with disable_logging=True.

    Without it the browser subprocess inherits our stdout and stderr, so every
    render dumps dbus connection errors and the browser's own "bytes written"
    notice into the user's console, burying the assistant output.
    """
    monkeypatch.setattr(
        html_to_png_base64, "get_storage_path", lambda: str(tmp_path)
    )

    mock_hti = mock.Mock()

    def fake_screenshot(html_str, save_as, size):
        (tmp_path / save_as).write_bytes(b"x")

    mock_hti.screenshot = fake_screenshot

    with mock.patch.object(html_to_png_base64, "html2image") as lazy:
        lazy.Html2Image.return_value = mock_hti
        html_to_png_base64.html_to_png_base64("<html></html>")

    lazy.Html2Image.assert_called_once_with(disable_logging=True)


def test_html_to_png_base64_returns_none_when_the_browser_is_missing(
    tmp_path, monkeypatch,
):
    """A machine without Chrome gets None instead of a FileNotFoundError.

    Constructing the wrapper discovers the browser executable, so an absent
    browser raises from the constructor. Callers fall back to showing the
    source, and a raise here would take down the run loop instead.
    """
    monkeypatch.setattr(
        html_to_png_base64, "get_storage_path", lambda: str(tmp_path)
    )

    with mock.patch.object(html_to_png_base64, "html2image") as lazy:
        lazy.Html2Image.side_effect = FileNotFoundError("chrome")
        result = html_to_png_base64.html_to_png_base64("<html></html>")

    assert result is None


def test_html_to_png_base64_returns_none_when_no_file_is_written(
    tmp_path, monkeypatch,
):
    """A render that produces no file returns None instead of raising on open.

    The browser can exit without writing the screenshot (snap confinement, a
    page that fails to load); the missing file used to raise FileNotFoundError
    from open() after a successful screenshot() call.
    """
    monkeypatch.setattr(
        html_to_png_base64, "get_storage_path", lambda: str(tmp_path)
    )

    mock_hti = mock.Mock()
    mock_hti.screenshot = mock.Mock()

    with mock.patch.object(html_to_png_base64, "html2image") as lazy:
        lazy.Html2Image.return_value = mock_hti
        result = html_to_png_base64.html_to_png_base64("<html></html>")

    assert result is None


def test_html_to_png_base64_returns_none_when_the_screenshot_cannot_be_read(
    tmp_path, monkeypatch,
):
    """A file that exists but cannot be opened returns None instead of raising.

    The exists() check passes when the browser leaves an unreadable file behind
    (permissions, a partial write), so the read itself is guarded; an OSError
    escaping this call would take down the run loop.
    """
    monkeypatch.setattr(
        html_to_png_base64, "get_storage_path", lambda: str(tmp_path)
    )

    mock_hti = mock.Mock()

    def fake_screenshot(html_str, save_as, size):
        (tmp_path / save_as).write_bytes(b"x")

    mock_hti.screenshot = fake_screenshot

    with (
        mock.patch.object(html_to_png_base64, "html2image") as lazy,
        mock.patch("builtins.open", side_effect=OSError("unreadable")),
    ):
        lazy.Html2Image.return_value = mock_hti
        result = html_to_png_base64.html_to_png_base64("<html></html>")

    assert result is None


def test_html_to_png_base64_returns_the_image_when_cleanup_fails(
    tmp_path, monkeypatch,
):
    """A failed temp-file deletion does not discard the rendered image.

    os.remove runs in a finally block, so deleting the leftover file is best
    effort: its OSError must not replace the base64 result the caller needs.
    """
    import base64

    monkeypatch.setattr(
        html_to_png_base64, "get_storage_path", lambda: str(tmp_path)
    )
    png_bytes = b"\x89PNG\r\n\x1a\nfake"

    mock_hti = mock.Mock()

    def fake_screenshot(html_str, save_as, size):
        (tmp_path / save_as).write_bytes(png_bytes)

    mock_hti.screenshot = fake_screenshot

    with (
        mock.patch.object(html_to_png_base64, "html2image") as lazy,
        mock.patch.object(
            html_to_png_base64.os, "remove", side_effect=OSError("busy")
        ),
    ):
        lazy.Html2Image.return_value = mock_hti
        result = html_to_png_base64.html_to_png_base64("<html></html>")

    assert result == base64.b64encode(png_bytes).decode()


def test_html_to_png_base64_retries_with_logging_when_the_png_is_missing(
    tmp_path, monkeypatch,
):
    """A missing PNG triggers one re-render with the browser's output enabled.

    html2image reports a failed capture only by not writing the file, so the
    quiet render leaves the user with no explanation. The retry lets Chromium's
    own diagnostics (missing shared libraries, a sandbox refusal) reach the
    console, and recovers the image when the first capture failed transiently.
    """
    import base64

    monkeypatch.setattr(
        html_to_png_base64, "get_storage_path", lambda: str(tmp_path)
    )
    png_bytes = b"\x89PNG\r\n\x1a\nretried"

    quiet_hti = mock.Mock()
    quiet_hti.screenshot = mock.Mock()

    loud_hti = mock.Mock()

    def fake_screenshot(html_str, save_as, size):
        (tmp_path / save_as).write_bytes(png_bytes)

    loud_hti.screenshot = fake_screenshot

    with mock.patch.object(html_to_png_base64, "html2image") as lazy:
        lazy.Html2Image.side_effect = [quiet_hti, loud_hti]
        result = html_to_png_base64.html_to_png_base64("<html></html>")

    assert result == base64.b64encode(png_bytes).decode()
    assert lazy.Html2Image.call_args_list == [
        mock.call(disable_logging=True),
        mock.call(disable_logging=False),
    ]


def test_html_to_png_base64_returns_none_when_the_logging_retry_also_misses(
    tmp_path, monkeypatch,
):
    """A second failed capture returns None rather than raising.

    The retry exists to surface the browser's diagnosis, not to change the
    contract: callers fall back to showing the source, and an exception here
    would take down the run loop.
    """
    monkeypatch.setattr(
        html_to_png_base64, "get_storage_path", lambda: str(tmp_path)
    )

    quiet_hti = mock.Mock()
    quiet_hti.screenshot = mock.Mock()
    loud_hti = mock.Mock()
    loud_hti.screenshot = mock.Mock()

    with mock.patch.object(html_to_png_base64, "html2image") as lazy:
        lazy.Html2Image.side_effect = [quiet_hti, loud_hti]
        result = html_to_png_base64.html_to_png_base64("<html></html>")

    assert result is None
    assert lazy.Html2Image.call_args_list == [
        mock.call(disable_logging=True),
        mock.call(disable_logging=False),
    ]


def test_html_to_png_base64_returns_none_when_the_logging_retry_raises(
    tmp_path, monkeypatch,
):
    """A retry that cannot launch the browser returns None instead of raising.

    The first render can succeed in constructing the wrapper and still write no
    file; the retry then rediscovers the executable and may fail outright (for
    example, Chrome was removed between the two attempts). Callers need the
    fallback path, not an exception from the diagnostic attempt.
    """
    monkeypatch.setattr(
        html_to_png_base64, "get_storage_path", lambda: str(tmp_path)
    )

    quiet_hti = mock.Mock()
    quiet_hti.screenshot = mock.Mock()

    with mock.patch.object(html_to_png_base64, "html2image") as lazy:
        lazy.Html2Image.side_effect = [quiet_hti, FileNotFoundError("chrome")]
        result = html_to_png_base64.html_to_png_base64("<html></html>")

    assert result is None

