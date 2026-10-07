"""Regression tests for #403: edge clamping must clip, not shrink.

Expanding a box near the right/bottom edge used to compute the clamped
width as image_width - x - width (re-subtracting the original size),
collapsing the click target instead of trimming it to the edge.
"""

import sys
import types
from unittest import mock

# point.py imports cv2 at module level, which is not available in CI.
# _expand_boxes is a pure function that doesn't use cv2, so mock the import
# with a proper module spec.
cv2_mock = types.ModuleType("cv2")
cv2_mock.__spec__ = mock.MagicMock()
with mock.patch.dict(sys.modules, {"cv2": cv2_mock}):
    from interpreter.core.computer.display.point.point import _expand_boxes


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
