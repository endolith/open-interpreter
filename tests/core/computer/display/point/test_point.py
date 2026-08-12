"""Characterization tests for ``computer.display.point``.

The upcoming ``computer/* -> toolbox/*`` port renames this module; these tests
trip when the import path or the dispatch/geometry behavior changes.
``point.py`` imports torch/sentence_transformers/timm/nltk/cv2 at module level,
so every test installs bare stubs via ``install_point_heavy_deps`` before the
module is first imported.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
from PIL import Image

from tests.helpers import install_point_heavy_deps


def _import_point(monkeypatch):
    install_point_heavy_deps(monkeypatch)
    import interpreter.core.computer.display.point.point as point_mod

    return point_mod


def test_point_routes_quoted_description_to_text_search(monkeypatch):
    """point() forwards a quoted description to find_text_in_image, not find_icon."""
    point_mod = _import_point(monkeypatch)

    screenshot = SimpleNamespace()
    with mock.patch.object(point_mod, "find_text_in_image", return_value=[(0.5, 0.5)]) as find_text:
        with mock.patch.object(point_mod, "find_icon") as find_icon:
            result = point_mod.point('"hello world"', screenshot, False, {})

    find_text.assert_called_once_with("hello world", screenshot, False)
    find_icon.assert_not_called()
    assert result == [(0.5, 0.5)]


def test_point_routes_unquoted_description_to_icon_search(monkeypatch):
    """point() forwards an unquoted description to find_icon."""
    point_mod = _import_point(monkeypatch)

    with mock.patch.object(point_mod, "find_icon", return_value=[(0.1, 0.2)]) as find_icon:
        result = point_mod.point("folder", None, True, {"some": "hashes"})

    find_icon.assert_called_once_with("folder", None, True, {"some": "hashes"})
    assert result == [(0.1, 0.2)]


def test_find_icon_filters_extremes_and_returns_normalized_center(monkeypatch):
    """find_icon() drops boxes outside the env-configured size range and returns
    the box center as fractional (x, y) coordinates of the screenshot size."""
    point_mod = _import_point(monkeypatch)
    monkeypatch.setenv("OI_POINT_MIN_ICON_WIDTH", "10")
    monkeypatch.setenv("OI_POINT_MAX_ICON_WIDTH", "500")
    monkeypatch.setenv("OI_POINT_MIN_ICON_HEIGHT", "10")
    monkeypatch.setenv("OI_POINT_MAX_ICON_HEIGHT", "500")
    monkeypatch.setenv("OI_POINT_PIXEL_EXPAND", "7")
    monkeypatch.setenv("OI_POINT_OVERLAP", "True")

    screenshot = Image.new("RGB", (200, 100), "white")
    boxes = [
        {"x": 20, "y": 20, "width": 30, "height": 30},
        {"x": 800, "y": 800, "width": 700, "height": 700},
        {"x": 0, "y": 0, "width": 5, "height": 5},
    ]

    captured = {}

    def fake_image_search(description, icons, hashes, debug):
        captured["description"] = description
        captured["icons"] = icons
        return icons[:1]

    with mock.patch.object(point_mod, "get_element_boxes", return_value=boxes):
        with mock.patch.object(
            point_mod, "pytesseract_get_text_bounding_boxes", return_value=[]
        ):
            with mock.patch.object(point_mod, "image_search", side_effect=fake_image_search):
                result = point_mod.find_icon("folder", screenshot, False, None)

    # Only the one valid box survives filtering.
    assert len(captured["icons"]) == 1
    icon = captured["icons"][0]
    assert icon["x"] == 13 and icon["y"] == 13
    assert icon["width"] == 44 and icon["height"] == 44
    # Center of the original box (20+15, 20+15) normalized to the 200x100 image.
    assert icon["coordinate"] == (0.175, 0.35)
    assert result == [(0.175, 0.35)]
    # find_icon appends " icon" when the description doesn't already mention it.
    assert captured["description"] == "folder icon"


def test_find_icon_skips_description_icon_suffix(monkeypatch):
    """find_icon() does not append another " icon" when the description already
    mentions "icon"."""
    point_mod = _import_point(monkeypatch)
    monkeypatch.setenv("OI_POINT_OVERLAP", "True")

    screenshot = Image.new("RGB", (200, 100), "white")
    captured = {}

    def fake_image_search(description, icons, hashes, debug):
        captured["description"] = description
        return []

    with mock.patch.object(point_mod, "get_element_boxes", return_value=[]):
        with mock.patch.object(
            point_mod, "pytesseract_get_text_bounding_boxes", return_value=[]
        ):
            with mock.patch.object(point_mod, "image_search", side_effect=fake_image_search):
                point_mod.find_icon("app icon", screenshot, False, None)

    assert captured["description"] == "app icon"


def test_find_icon_filters_boxes_overlapping_text(monkeypatch):
    """find_icon() drops icon boxes that overlap a real-word text block detected
    by tesseract (so text is not mistaken for an icon)."""
    point_mod = _import_point(monkeypatch)
    monkeypatch.setattr(point_mod, "english_words", {"hello"})
    monkeypatch.setenv("OI_POINT_OVERLAP", "True")

    screenshot = Image.new("RGB", (200, 100), "white")
    boxes = [{"x": 20, "y": 20, "width": 30, "height": 30}]
    text_blocks = [
        {"left": 10, "top": 10, "width": 50, "height": 50, "text": "hello"}
    ]

    with mock.patch.object(point_mod, "get_element_boxes", return_value=boxes):
        with mock.patch.object(
            point_mod, "pytesseract_get_text_bounding_boxes", return_value=text_blocks
        ):
            with mock.patch.object(point_mod, "image_search", return_value=[]) as search:
                result = point_mod.find_icon("folder", screenshot, False, None)

    search.assert_called_once()
    assert search.call_args[0][1] == []
    assert result == []


def test_find_icon_combines_overlapping_boxes(monkeypatch):
    """find_icon() merges overlapping icon boxes into one and returns its center."""
    point_mod = _import_point(monkeypatch)
    monkeypatch.setenv("OI_POINT_PIXEL_EXPAND", "7")
    monkeypatch.setenv("OI_POINT_OVERLAP", "True")

    screenshot = Image.new("RGB", (200, 100), "white")
    boxes = [
        {"x": 10, "y": 10, "width": 20, "height": 20},
        {"x": 25, "y": 25, "width": 20, "height": 20},
    ]

    def fake_image_search(description, icons, hashes, debug):
        return icons[:1]

    with mock.patch.object(point_mod, "get_element_boxes", return_value=boxes):
        with mock.patch.object(
            point_mod, "pytesseract_get_text_bounding_boxes", return_value=[]
        ):
            with mock.patch.object(point_mod, "image_search", side_effect=fake_image_search):
                result = point_mod.find_icon("folder", screenshot, False, None)

    assert result == [(0.1, 0.2)]


@pytest.mark.darwin_ci
def test_take_screenshot_to_pil_captures_and_cleans_up(monkeypatch, tmp_path):
    """point's take_screenshot_to_pil() runs `screencapture -x` (a macOS-only
    command), loads the PNG it produced, and removes the temporary file. The
    command contract is macOS-specific, so the test runs in the macOS CI lane."""
    point_mod = _import_point(monkeypatch)

    filename = str(tmp_path / "temp_screenshot.png")
    Image.new("RGB", (50, 50), "blue").save(filename)

    with mock.patch.object(point_mod.subprocess, "run") as run:
        with mock.patch.object(point_mod.os, "remove") as remove:
            result = point_mod.take_screenshot_to_pil(filename=filename)

    run.assert_called_once_with(["screencapture", "-x", filename], check=True)
    remove.assert_called_once_with(filename)
    assert result.size == (50, 50)
