"""Characterization tests for ``computer.display``.

These mock pyautogui/pywinctl/screeninfo/requests so the tests run headless in
CI. They exist to trip loudly when the ``computer/* -> toolbox/*`` port renames
or reworks the display subsystem.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
from PIL import Image

from tests.helpers import install_point_heavy_deps


def _make_display(computer=None):
    from interpreter.core.computer.display.display import Display

    return Display(computer or SimpleNamespace())


def _make_offline_computer(**overrides):
    attrs = dict(
        debug=False,
        offline=True,
        api_base="http://example.com:8080",
        emit_images=False,
        verbose=False,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _patch_pyautogui(**attrs):
    """Patch the module-level ``pyautogui`` name (it is ``None`` via lazy_import
    on machines without the package) with a mock and return it."""
    pyautogui = mock.Mock()
    for name, value in attrs.items():
        getattr(pyautogui, name).return_value = value
    patcher = mock.patch(
        "interpreter.core.computer.display.display.pyautogui", pyautogui
    )
    return patcher, pyautogui


def test_size_delegates_to_pyautogui():
    """Display.size() forwards to pyautogui.size()."""
    display = _make_display()
    patcher, pyautogui = _patch_pyautogui(size=(1920, 1080))
    with patcher:
        assert display.size() == (1920, 1080)
    pyautogui.size.assert_called_once_with()


def test_width_and_height_properties_lazily_query_pyautogui():
    """Display.width/height fetch pyautogui.size() once per property, then cache."""
    display = _make_display()
    patcher, pyautogui = _patch_pyautogui(size=(800, 600))
    with patcher:
        assert display.width == 800
        assert display.width == 800  # cached
        assert display.height == 600
        assert display.height == 600  # cached
    # Two distinct size() calls: one for the first width access, one for height.
    assert pyautogui.size.call_count == 2


def test_center_uses_width_and_height():
    """Display.center() returns the midpoints of width and height."""
    display = _make_display()
    patcher, _ = _patch_pyautogui(size=(100, 80))
    with patcher:
        assert display.center() == (50, 40)


def test_info_returns_get_displays():
    """Display.info() delegates to get_displays()."""
    display = _make_display()
    monitors = [SimpleNamespace(name="A"), SimpleNamespace(name="B")]
    with mock.patch(
        "interpreter.core.computer.display.display.get_displays",
        return_value=monitors,
    ) as get_displays:
        assert display.info() == monitors
    get_displays.assert_called_once_with()


def test_view_delegates_to_screenshot():
    """Display.view() forwards all its parameters to screenshot()."""
    display = _make_display()
    with mock.patch.object(display, "screenshot", return_value="pil_image") as screenshot:
        result = display.view(show=False, quadrant=2, screen=1, combine_screens=False, active_app_only=False)
    screenshot.assert_called_once_with(
        screen=1, show=False, quadrant=2, combine_screens=False, active_app_only=False
    )
    assert result == "pil_image"


def test_screenshot_captures_active_window_region():
    """Display.screenshot() captures just the active window when active_app_only."""
    display = _make_display()
    active_window = SimpleNamespace(left=10, top=20, width=100, height=200)
    pil = Image.new("RGBA", (100, 200), "red")
    patcher, pyautogui = _patch_pyautogui(screenshot=pil)
    pywinctl = mock.Mock()
    pywinctl.getActiveWindow.return_value = active_window
    with mock.patch("interpreter.core.computer.display.display.pywinctl", pywinctl):
        with patcher:
            result = display.screenshot(show=False)

    pyautogui.screenshot.assert_called_once_with(region=(10, 20, 100, 200))
    assert result.mode == "RGB"


def test_screenshot_falls_back_to_full_screen_without_active_window():
    """Display.screenshot() captures the full screen when there is no active window."""
    display = _make_display()
    pil = Image.new("RGB", (50, 50))
    patcher, pyautogui = _patch_pyautogui(screenshot=pil)
    pywinctl = mock.Mock()
    pywinctl.getActiveWindow.return_value = None
    with mock.patch("interpreter.core.computer.display.display.pywinctl", pywinctl):
        with patcher:
            display.screenshot(show=False)

    pyautogui.screenshot.assert_called_once_with()


def test_screenshot_all_screens_uses_take_screenshot_to_pil():
    """Display.screenshot(active_app_only=False) delegates to take_screenshot_to_pil()."""
    display = _make_display()
    pil = Image.new("RGB", (50, 50))
    with mock.patch(
        "interpreter.core.computer.display.display.take_screenshot_to_pil",
        return_value=pil,
    ) as take:
        result = display.screenshot(show=False, active_app_only=False)

    take.assert_called_once_with(screen=0, combine_screens=True)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"


def test_screenshot_quadrant_regions():
    """Display.screenshot(quadrant=N) captures the matching quarter of the screen."""
    display = _make_display()
    patcher, pyautogui = _patch_pyautogui(size=(200, 200), screenshot=Image.new("RGB", (100, 100)))
    with patcher:
        display.screenshot(show=False, quadrant=1)
        display.screenshot(show=False, quadrant=4)

    pyautogui.screenshot.assert_any_call(region=(0, 0, 100, 100))
    pyautogui.screenshot.assert_any_call(region=(100, 100, 100, 100))


def test_screenshot_invalid_quadrant_raises():
    """Display.screenshot() rejects quadrants outside 1-4."""
    display = _make_display()
    patcher, _ = _patch_pyautogui(size=(200, 200))
    with patcher:
        with pytest.raises(ValueError, match="Invalid quadrant"):
            display.screenshot(show=False, quadrant=5)


def test_find_quoted_delegates_to_find_text():
    """Display.find() routes a quoted description to find_text()."""
    display = _make_display()
    with mock.patch.object(display, "find_text", return_value=[(0.5, 0.5)]) as find_text:
        result = display.find('"hello world"', screenshot="img")

    find_text.assert_called_once_with("hello world", "img")
    assert result == [(0.5, 0.5)]


def test_find_unquoted_uses_point(monkeypatch):
    """Display.find() routes an unquoted description through point()."""
    install_point_heavy_deps(monkeypatch)
    computer = _make_offline_computer()
    display = _make_display(computer)

    with mock.patch(
        "interpreter.core.computer.display.point.point.point",
        return_value=[(0.1, 0.2)],
    ) as point:
        result = display.find("folder", screenshot="img")

    point.assert_called_once_with("folder", "img", False, {})
    assert result == [(0.1, 0.2)]


def test_find_raises_when_offline_and_point_fails(monkeypatch):
    """Display.find() re-raises point failures when offline."""
    install_point_heavy_deps(monkeypatch)
    computer = _make_offline_computer()
    display = _make_display(computer)

    with mock.patch(
        "interpreter.core.computer.display.point.point.point",
        side_effect=ValueError("boom"),
    ):
        with pytest.raises(ValueError, match="boom"):
            display.find("folder", screenshot="img")


def test_find_falls_back_to_icon_api_when_point_fails(monkeypatch):
    """Display.find() retries via the remote /point/ API when point() fails and
    the computer is online."""
    install_point_heavy_deps(monkeypatch)
    computer = _make_offline_computer(offline=False)
    display = _make_display(computer)

    screenshot = Image.new("RGB", (800, 600))
    response = SimpleNamespace(json=lambda: {"ok": True})
    with mock.patch(
        "interpreter.core.computer.display.point.point.point",
        side_effect=ValueError("boom"),
    ):
        with mock.patch(
            "interpreter.core.computer.display.display.requests.post",
            return_value=response,
        ) as post:
            result = display.find("folder", screenshot=screenshot)

    assert result == {"ok": True}
    assert post.call_args[0][0] == "http://example.com:8080/point/"
    assert post.call_args[1]["json"]["query"] == "folder"
    assert post.call_args[1]["json"]["base64"]


def test_find_text_offline_uses_local_vision():
    """Display.find_text() falls back to local find_text_in_image when offline."""
    computer = _make_offline_computer()
    display = _make_display(computer)

    with mock.patch(
        "interpreter.core.computer.display.display.find_text_in_image",
        return_value=[(0.25, 0.75)],
    ) as find_text_in_image:
        result = display.find_text("hello", screenshot="img")

    find_text_in_image.assert_called_once_with("img", "hello", False)
    assert result == [{"coordinates": (0.25, 0.75), "text": "", "similarity": 1}]


def test_find_text_online_uses_remote_api():
    """Display.find_text() posts the screenshot to the remote /point/text/ API when
    online."""
    computer = _make_offline_computer(offline=False)
    display = _make_display(computer)

    screenshot = Image.new("RGB", (10, 10))
    response = SimpleNamespace(json=lambda: {"text": [["hello"]]})
    with mock.patch(
        "interpreter.core.computer.display.display.requests.post",
        return_value=response,
    ) as post:
        result = display.find_text("hello", screenshot=screenshot)

    assert result == {"text": [["hello"]]}
    assert post.call_args[0][0] == "http://example.com:8080/point/text/"
    assert post.call_args[1]["json"]["query"] == "hello"


def test_get_text_offline_uses_pytesseract():
    """Display.get_text_as_list_of_lists() uses pytesseract locally when offline."""
    computer = _make_offline_computer()
    display = _make_display(computer)

    with mock.patch(
        "interpreter.core.computer.display.display.pytesseract_get_text",
        return_value="hello",
    ) as pytesseract_get_text:
        result = display.get_text_as_list_of_lists(screenshot="img")

    pytesseract_get_text.assert_called_once_with("img")
    assert result == "hello"


def _patch_screeninfo(monitors):
    screeninfo = mock.Mock()
    screeninfo.get_monitors.return_value = monitors
    return mock.patch(
        "interpreter.core.computer.display.display.screeninfo", screeninfo
    )


def test_take_screenshot_to_pil_primary_monitor():
    """take_screenshot_to_pil(screen=0) captures the primary monitor's region."""
    from interpreter.core.computer.display.display import take_screenshot_to_pil

    monitors = [SimpleNamespace(x=0, y=0, width=800, height=600)]
    pil = Image.new("RGB", (800, 600))
    patcher, pyautogui = _patch_pyautogui(screenshot=pil)
    with _patch_screeninfo(monitors):
        with patcher:
            result = take_screenshot_to_pil(screen=0)

    pyautogui.screenshot.assert_called_once_with(region=(0, 0, 800, 600))
    assert result is pil


def test_take_screenshot_to_pil_secondary_monitor():
    """take_screenshot_to_pil(screen=N) captures the Nth monitor's region."""
    from interpreter.core.computer.display.display import take_screenshot_to_pil

    monitors = [
        SimpleNamespace(x=0, y=0, width=800, height=600),
        SimpleNamespace(x=800, y=0, width=1024, height=768),
    ]
    pil = Image.new("RGB", (1024, 768))
    patcher, pyautogui = _patch_pyautogui(screenshot=pil)
    with _patch_screeninfo(monitors):
        with patcher:
            result = take_screenshot_to_pil(screen=1)

    pyautogui.screenshot.assert_called_once_with(region=(800, 0, 1024, 768))
    assert result is pil


def test_take_screenshot_to_pil_all_combines_into_single_image():
    """take_screenshot_to_pil(screen=-1, combine_screens=True) stitches all monitors
    into one image with OpenCV."""
    import numpy as np

    from interpreter.core.computer.display.display import take_screenshot_to_pil

    monitors = [
        SimpleNamespace(x=0, y=0, width=20, height=10),
        SimpleNamespace(x=20, y=0, width=20, height=10),
    ]
    cv2 = mock.Mock()
    cv2.cvtColor.side_effect = lambda img, code: img
    cv2.getTextSize.return_value = ((100, 20), 0)

    patcher, pyautogui = _patch_pyautogui(screenshot=Image.new("RGB", (20, 10), "red"))
    with _patch_screeninfo(monitors):
        with patcher:
            with mock.patch("interpreter.core.computer.display.display.cv2", cv2):
                result = take_screenshot_to_pil(screen=-1, combine_screens=True)

    assert isinstance(result, Image.Image)
    assert result.size == (40, 10)
    assert np.any(np.asarray(result))
    cv2.putText.assert_called()
    # getTextSize runs twice per monitor (once for the initial scale, once after
    # the font scale is recalculated to fit the monitor).
    assert cv2.getTextSize.call_count == 4


def test_take_screenshot_to_pil_all_returns_list_without_combine():
    """take_screenshot_to_pil(screen=-1, combine_screens=False) returns one PIL
    image per monitor."""
    from interpreter.core.computer.display.display import take_screenshot_to_pil

    monitors = [
        SimpleNamespace(x=0, y=0, width=20, height=10),
        SimpleNamespace(x=20, y=0, width=20, height=10),
    ]
    patcher, pyautogui = _patch_pyautogui(screenshot=Image.new("RGB", (20, 10), "red"))
    with _patch_screeninfo(monitors):
        with patcher:
            result = take_screenshot_to_pil(screen=-1, combine_screens=False)

    assert isinstance(result, list)
    assert len(result) == 2
    assert pyautogui.screenshot.call_count == 2
