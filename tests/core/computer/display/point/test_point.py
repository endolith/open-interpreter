"""Characterization tests for ``computer.display.point``.

These tests pin the current import path and dispatch/geometry behavior so any
rename or refactor trips loudly on drift. ``point.py`` imports
torch/sentence_transformers/timm/nltk/cv2 at module level, so every test
installs bare stubs via ``install_point_heavy_deps`` before the module is first
imported.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
from PIL import Image

import sys

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


def _fake_embed():
    """A tensor-ish fake with the .to/.unsqueeze surface image_search touches."""

    def _to(device):
        return _fake_embed()

    def _unsqueeze(_dim):
        return _fake_embed()

    e = SimpleNamespace(label="embed")
    e.to = _to
    e.unsqueeze = _unsqueeze
    return e


class _FakeBatch:
    """Mimics the tensor surface model.encode exposes: [0], [1:] and .to()."""

    def __init__(self, items):
        self._items = items

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return _FakeBatch(self._items[idx])
        return self._items[idx]

    def to(self, device):
        return self


def test_image_search_embeds_unhashed_and_filters_by_score(monkeypatch):
    """image_search embeds query + uncached icons, caches hashes, and returns
    only icons whose semantic score exceeds 90."""
    point_mod = _import_point(monkeypatch)

    icons = [
        {"hash": "new1", "data": "img1"},
        {"hash": "cached", "data": "img2"},
    ]
    hashes = {"cached": _fake_embed()}

    query = _fake_embed()
    model = mock.Mock()
    model.encode.return_value = _FakeBatch([query, _fake_embed()])
    monkeypatch.setattr(point_mod, "model", model)
    monkeypatch.setattr(
        point_mod.torch,
        "cat",
        lambda *_a, **_k: [_fake_embed(), hashes["cached"]],
    )
    monkeypatch.setattr(
        point_mod.util,
        "semantic_search",
        lambda *_a, **_k: [
            [
                {"corpus_id": 0, "score": 99.0},
                {"corpus_id": 1, "score": 90.0},
            ]
        ],
    )

    result = point_mod.image_search(query, icons, hashes, False)

    encoded = model.encode.call_args[0][0]
    assert encoded[0] is query
    assert encoded[1:] == ["img1"]
    assert "new1" in hashes
    # Only the strictly->90 hit survives; the 90.0 boundary is filtered out.
    assert [i["hash"] for i in result] == ["new1"]


def test_image_search_forces_top_hit_into_results(monkeypatch):
    """A low-scoring top hit is still included ahead of the qualifying ones."""
    point_mod = _import_point(monkeypatch)

    icons = [
        {"hash": "new1", "data": "img1"},
        {"hash": "new2", "data": "img2"},
    ]
    hashes = {}
    query = _fake_embed()
    model = mock.Mock()
    model.encode.return_value = _FakeBatch([query, _fake_embed(), _fake_embed()])
    monkeypatch.setattr(point_mod, "model", model)
    monkeypatch.setattr(
        point_mod.torch,
        "cat",
        lambda *_a, **_k: [_fake_embed(), _fake_embed()],
    )
    monkeypatch.setattr(
        point_mod.util,
        "semantic_search",
        lambda *_a, **_k: [
            [
                {"corpus_id": 0, "score": 50.0},
                {"corpus_id": 1, "score": 99.0},
            ]
        ],
    )

    result = point_mod.image_search(query, icons, hashes, False)

    assert [i["hash"] for i in result] == ["new1", "new2"]


def test_image_search_slow_model_uses_embed_images(monkeypatch):
    """When fast_model is False, embedding goes through embed_images()."""
    point_mod = _import_point(monkeypatch)
    monkeypatch.setattr(point_mod, "fast_model", False)

    icons = [{"hash": "new1", "data": "img1"}]
    hashes = {}
    embeds = _FakeBatch([_fake_embed(), _fake_embed()])
    monkeypatch.setattr(
        point_mod, "embed_images", mock.Mock(return_value=embeds), raising=False
    )
    monkeypatch.setattr(point_mod, "transforms", None, raising=False)
    monkeypatch.setattr(point_mod.torch, "cat", lambda *_a, **_k: [_fake_embed()])
    monkeypatch.setattr(
        point_mod.util,
        "semantic_search",
        lambda *_a, **_k: [[{"corpus_id": 0, "score": 100.0}]],
    )

    result = point_mod.image_search("folder", icons, hashes, False)

    assert [i["hash"] for i in result] == ["new1"]


def test_get_element_boxes_builds_boxes_from_contours(monkeypatch):
    """get_element_boxes runs the cv2 pipeline and returns boundingRect boxes."""
    point_mod = _import_point(monkeypatch)

    screenshot = Image.new("RGB", (100, 100), "white")

    monkeypatch.setattr(point_mod.cv2, "cvtColor", lambda *_a, **_k: "bgr", raising=False)
    monkeypatch.setattr(
        point_mod.cv2, "adaptiveThreshold", lambda *_a, **_k: "binary", raising=False
    )
    monkeypatch.setattr(
        point_mod.cv2,
        "findContours",
        lambda *_a, **_k: ([{"contour": 1}, {"contour": 2}], None),
        raising=False,
    )
    monkeypatch.setattr(point_mod.cv2, "drawContours", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        point_mod.cv2,
        "boundingRect",
        lambda contour: {"contour": 1} == contour and (5, 10, 20, 30) or (15, 25, 10, 5),
        raising=False,
    )
    # process_image's default args reference these at definition time.
    monkeypatch.setattr(point_mod.cv2, "ADAPTIVE_THRESH_MEAN_C", 0, raising=False)
    monkeypatch.setattr(point_mod.cv2, "THRESH_BINARY_INV", 1, raising=False)
    monkeypatch.setattr(point_mod.cv2, "COLOR_RGB2BGR", 4, raising=False)
    monkeypatch.setattr(point_mod.cv2, "COLOR_BGR2GRAY", 5, raising=False)
    monkeypatch.setattr(point_mod.cv2, "RETR_LIST", 6, raising=False)
    monkeypatch.setattr(point_mod.cv2, "CHAIN_APPROX_NONE", 7, raising=False)

    boxes = point_mod.get_element_boxes(screenshot, False)

    assert boxes == [
        {"x": 5, "y": 10, "width": 20, "height": 30},
        {"x": 15, "y": 25, "width": 10, "height": 5},
    ]


def test_get_element_boxes_permutates_when_env_set(monkeypatch):
    """OI_POINT_PERMUTATE=True varies threshold parameters across iterations."""
    import types

    point_mod = _import_point(monkeypatch)
    monkeypatch.setenv("OI_POINT_PERMUTATE", "True")

    screenshot = Image.new("RGB", (100, 100), "white")

    monkeypatch.setattr(point_mod.cv2, "cvtColor", lambda *_a, **_k: "bgr", raising=False)
    adaptive = mock.Mock(return_value="binary")
    monkeypatch.setattr(point_mod.cv2, "adaptiveThreshold", adaptive, raising=False)
    monkeypatch.setattr(
        point_mod.cv2,
        "findContours",
        lambda *_a, **_k: ([{"contour": 1}], None),
        raising=False,
    )
    monkeypatch.setattr(point_mod.cv2, "drawContours", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(point_mod.cv2, "boundingRect", lambda _c: (1, 2, 3, 4), raising=False)
    # process_image's default args reference these at definition time.
    monkeypatch.setattr(point_mod.cv2, "ADAPTIVE_THRESH_MEAN_C", 0, raising=False)
    monkeypatch.setattr(point_mod.cv2, "THRESH_BINARY_INV", 1, raising=False)
    monkeypatch.setattr(point_mod.cv2, "ADAPTIVE_THRESH_GAUSSIAN_C", 2, raising=False)
    monkeypatch.setattr(point_mod.cv2, "THRESH_BINARY", 3, raising=False)
    monkeypatch.setattr(point_mod.cv2, "COLOR_RGB2BGR", 4, raising=False)
    monkeypatch.setattr(point_mod.cv2, "COLOR_BGR2GRAY", 5, raising=False)
    monkeypatch.setattr(point_mod.cv2, "RETR_LIST", 6, raising=False)
    monkeypatch.setattr(point_mod.cv2, "CHAIN_APPROX_NONE", 7, raising=False)

    # get_element_boxes does `import random` locally, so stub sys.modules.
    random_stub = types.ModuleType("random")
    random_stub.uniform = mock.Mock(
        side_effect=[float(n) for n in range(1, 11)]
    )
    # Select from real production candidate sets: blockSize (odd 1..9),
    # adaptiveMethod, thresholdType. Repeat the sequence so all 10 iterations
    # are covered; each iteration still varies which candidate is picked.
    random_stub.choice = mock.Mock(
        side_effect=[1, 0, 3, 5, 2, 1, 9, 0, 3, 1, 2, 1] * 3
    )
    random_stub.randint = mock.Mock(side_effect=[-5, 5] * 5)
    monkeypatch.setitem(sys.modules, "random", random_stub)

    boxes = point_mod.get_element_boxes(screenshot, False)

    assert boxes == [{"x": 1, "y": 2, "width": 3, "height": 4}]
    assert adaptive.call_count == 10
    param_sets = {
        (c.kwargs["adaptiveMethod"], c.kwargs["thresholdType"], c.kwargs["C"])
        for c in adaptive.call_args_list
    }
    # The random draws must have actually changed the threshold parameters.
    assert len(param_sets) > 1
