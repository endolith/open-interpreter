"""Characterization tests for ``computer.utils.computer_vision``.

cv2 and pytesseract are optional lazy imports (not installed in CI), so each
test patches them with mocks; numpy and PIL are real. These pin the text
detection / OCR helper behavior used by ``computer.display`` and
``computer.display.point``.
"""

from unittest import mock

import pytest
from PIL import Image

from interpreter.core.computer.utils import computer_vision as cv_mod


def _image(width=200, height=100):
    image = Image.new("RGB", (width, height), "white")
    image.format = "PNG"
    return image


def _patch_cv2():
    cv2 = mock.Mock()
    cv2.cvtColor.side_effect = lambda image, code: image
    return mock.patch.object(cv_mod, "cv2", cv2), cv2


def _patch_pytesseract(data):
    pytesseract = mock.Mock()
    pytesseract.Output.DICT = "dict"
    pytesseract.image_to_data.return_value = data
    return mock.patch.object(cv_mod, "pytesseract", pytesseract), pytesseract


def test_pytesseract_get_text_delegates_to_image_to_string():
    """pytesseract_get_text() forwards the image to pytesseract.image_to_string."""
    pytesseract = mock.Mock()
    pytesseract.image_to_string.return_value = "hello"
    with mock.patch.object(cv_mod, "pytesseract", pytesseract):
        image = _image()
        result = cv_mod.pytesseract_get_text(image)

    pytesseract.image_to_string.assert_called_once_with(image)
    assert result == "hello"


def test_pytesseract_get_text_raises_when_module_missing():
    """pytesseract_get_text() raises ImportError when pytesseract isn't installed."""
    with mock.patch.object(cv_mod, "pytesseract", None):
        with pytest.raises(ImportError, match="pytesseract"):
            cv_mod.pytesseract_get_text(_image())


def test_pytesseract_get_text_bounding_boxes_builds_box_dicts():
    """pytesseract_get_text_bounding_boxes() converts pytesseract's parallel lists
    into one dict per detected text box."""
    cv2_patcher, _ = _patch_cv2()
    data = {
        "text": ["a", "b"],
        "top": [1, 2],
        "left": [3, 4],
        "width": [5, 6],
        "height": [7, 8],
    }
    pytesseract_patcher, _ = _patch_pytesseract(data)
    with cv2_patcher, pytesseract_patcher:
        boxes = cv_mod.pytesseract_get_text_bounding_boxes(_image())

    assert boxes == [
        {"text": "a", "top": 1, "left": 3, "width": 5, "height": 7},
        {"text": "b", "top": 2, "left": 4, "width": 6, "height": 8},
    ]


def test_find_text_in_image_returns_normalized_matching_center():
    """find_text_in_image() narrows the box to the matching substring, returns its
    center normalized to image size, and draws the box."""
    cv2_patcher, cv2 = _patch_cv2()
    data = {
        "level": [3, 3],
        "text": ["hi there", "other"],
        "left": [10, 50],
        "top": [20, 60],
        "width": [80, 30],
        "height": [20, 15],
    }
    pytesseract_patcher, _ = _patch_pytesseract(data)
    with cv2_patcher, pytesseract_patcher:
        centers = cv_mod.find_text_in_image(_image(), "there")

    # Box narrowed to "there" inside "hi there": left 10+30, width 50.
    assert centers == [(0.325, 0.3)]
    assert cv2.rectangle.called
    assert cv2.putText.called


def test_find_text_in_image_word_fallback_pairs_word_centers():
    """find_text_in_image() falls back to pairing the centers of individually
    matched words when the whole phrase isn't found in one box."""
    cv2_patcher, _ = _patch_cv2()
    data = {
        "level": [3, 3],
        "text": ["alpha", "beta"],
        "left": [10, 100],
        "top": [20, 20],
        "width": [80, 80],
        "height": [20, 20],
    }
    pytesseract_patcher, _ = _patch_pytesseract(data)
    with cv2_patcher, pytesseract_patcher:
        centers = cv_mod.find_text_in_image(_image(), "alpha beta")

    # Midpoint of the two word centers, normalized: ((25+70)/2, 15) / (200, 100).
    assert centers == [(0.2375, 0.15)]


def test_find_text_in_image_debug_draws_each_box():
    """find_text_in_image(debug=True) draws and labels every detected box — not
    just the matching one — and still returns the normalized match center."""
    cv2_patcher, cv2 = _patch_cv2()
    data = {
        "level": [2, 2],
        "text": ["abc", "xyz"],
        "left": [0, 40],
        "top": [0, 30],
        "width": [10, 10],
        "height": [10, 10],
    }
    pytesseract_patcher, _ = _patch_pytesseract(data)
    with cv2_patcher, pytesseract_patcher:
        centers = cv_mod.find_text_in_image(_image(), "abc", debug=True)

    assert centers == [(0.025, 0.05)]
    put_texts = [call.args[1] for call in cv2.putText.call_args_list]
    assert "abc" in put_texts
    # The non-matching box is only labelled in debug mode.
    assert "xyz" in put_texts
    assert cv2.rectangle.called
