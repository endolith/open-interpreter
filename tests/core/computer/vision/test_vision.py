from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.vision.vision import Vision


def test_ocr_from_base64_lmc_uses_easyocr():
    """Vision.ocr() on a base64 LMC image should decode and run easyocr.readtext.

    We mock easyocr so CI does not download models or need a display. The test
    checks that the recognized text from readtext is returned unchanged.
    """
    import base64

    # Minimal valid 1x1 PNG (content does not matter; easyocr is mocked).
    png = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode()
    vision = Vision(computer=SimpleNamespace(debug=False))
    lmc = {"format": "base64.png", "content": png}

    fake_reader = mock.Mock()
    fake_reader.readtext.return_value = [(None, "OCR TEXT", None)]

    with mock.patch.object(vision, "load"):
        vision.easyocr = fake_reader
        assert vision.ocr(lmc=lmc) == "OCR TEXT"
    fake_reader.readtext.assert_called_once()
