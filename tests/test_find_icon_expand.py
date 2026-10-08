"""Regression tests for #403: edge clamping must clip, not shrink.

Expanding a box near the right/bottom edge used to compute the clamped
width as image_width - x - width (re-subtracting the original size),
collapsing the click target instead of trimming it to the edge.
"""

import re
import sys
from unittest import mock

# point.py imports cv2, nltk (and nltk.corpus), torch and sentence_transformers
# at module level, and its sibling computer_vision pulls in timm/pytesseract.
# Those ship in the optional `os`/`local` extras, which the base CI install does
# not pull in — importing point.py straight from this test used to abort the
# whole collection with ModuleNotFoundError. _expand_boxes is a pure function
# that touches none of them, so stub them unconditionally: presence is not a
# good test here either, since a broken install (e.g. an nltk whose scipy does
# not load) fails just as hard as an absent one.
_OPTIONAL_IMPORTS = (
    "cv2",
    "nltk",
    "nltk.corpus",
    "torch",
    "torchvision",
    "timm",
    "sentence_transformers",
    "transformers",
    "easyocr",
    "pytesseract",
    "screeninfo",
    "pywinctl",
    "plyer",
    "ipywidgets",
)
_MAX_STUB_ATTEMPTS = 40


def _stub_module(name):
    # A MagicMock tolerates the `from x import y` forms point.py and its
    # siblings use, which a bare ModuleType stub cannot.
    module = mock.MagicMock(name=name)
    module.__name__ = name
    module.__spec__ = mock.MagicMock()
    return module


def _import_expand_boxes():
    """Import _expand_boxes with the optional display extras stubbed out.

    Anything that turns out to be imported transitively but is not in the
    list above is added on the fly, so a future import does not break
    collection again.
    """
    saved = {}
    for name in _OPTIONAL_IMPORTS:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = _stub_module(name)

    try:
        for _ in range(_MAX_STUB_ATTEMPTS):
            try:
                from interpreter.core.computer.display.point.point import (
                    _expand_boxes,
                )

                return _expand_boxes
            except ImportError as exc:
                match = re.search(r"No module named '([^']+)'", str(exc))
                if match is None:
                    raise
                missing = match.group(1)
                if missing in saved:
                    raise
                saved[missing] = sys.modules.get(missing)
                sys.modules[missing] = _stub_module(missing)
        raise AssertionError("could not import _expand_boxes")
    finally:
        # Remove the stubs so they cannot leak into other tests, leaving real
        # modules (and the point module itself) in place.
        for name, original in saved.items():
            if isinstance(sys.modules.get(name), mock.MagicMock):
                if original is None:
                    del sys.modules[name]
                else:
                    sys.modules[name] = original


_expand_boxes = _import_expand_boxes()


def test_right_edge_box_clips_to_edge():
    """A box touching the right edge keeps covering the icon.

    200px-wide image, expand 7, box x=170 w=30: x shifts to 163 and the
    width clips to 200 - 163 = 37 instead of collapsing to 7.
    """
    boxes = [{"x": 170, "y": 50, "width": 30, "height": 30}]

    result = _expand_boxes(boxes, 200, 200, 7)

    assert result[0]["x"] == 163
    assert result[0]["width"] == 37
    assert result[0]["x"] + result[0]["width"] == 200


def test_bottom_edge_box_clips_to_edge():
    """Same clipping on the bottom edge."""
    boxes = [{"x": 50, "y": 170, "width": 30, "height": 30}]

    result = _expand_boxes(boxes, 200, 200, 7)

    assert result[0]["y"] == 163
    assert result[0]["height"] == 37
    assert result[0]["y"] + result[0]["height"] == 200


def test_interior_box_expands_fully():
    """A box clear of the edges grows by expand on every side."""
    boxes = [{"x": 50, "y": 50, "width": 30, "height": 30}]

    result = _expand_boxes(boxes, 200, 200, 7)

    assert result[0] == {"x": 43, "y": 43, "width": 44, "height": 44}
