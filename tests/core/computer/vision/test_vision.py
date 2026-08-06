"""Characterization tests for ``computer.vision``.

easyocr and the moondream transformers models are mocked or stubbed in
``sys.modules`` so no real vision models are loaded or downloaded.
"""

import base64
import io
import os
from types import SimpleNamespace
from unittest import mock

import pytest
from PIL import Image

from interpreter.core.computer.vision import vision as vision_mod
from interpreter.core.computer.vision.vision import Vision


def _make_vision():
    return Vision(SimpleNamespace(debug=False))


def _png_base64():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _easyocr_mock():
    easyocr = mock.Mock()
    easyocr.readtext.return_value = [[[1, 2, 3], "hello", 0.9], [[1, 2, 3], "world", 0.9]]
    return easyocr


def test_ocr_reads_text_from_path():
    """Vision.ocr(path=...) runs easyocr on the file and joins the words."""
    vision = _make_vision()
    vision.easyocr = _easyocr_mock()

    result = vision.ocr(path="img.png")

    assert result == "hello world"
    vision.easyocr.readtext.assert_called_once_with("img.png")


def test_ocr_decodes_base64_into_temp_file():
    """Vision.ocr(base_64=...) writes the decoded bytes to a temp PNG first."""
    vision = _make_vision()
    vision.easyocr = _easyocr_mock()

    read_path = None
    try:
        result = vision.ocr(base_64=_png_base64())
        read_path = vision.easyocr.readtext.call_args[0][0]
        assert result == "hello world"
        assert read_path.endswith(".png")
        assert os.path.exists(read_path)
    finally:
        if read_path:
            os.remove(read_path)


def test_ocr_accepts_lmc_path_format():
    """Vision.ocr(lmc={'format': 'path'}) uses the message's content as the path."""
    vision = _make_vision()
    vision.easyocr = _easyocr_mock()

    vision.ocr(lmc={"format": "path", "content": "/tmp/x.png"})

    vision.easyocr.readtext.assert_called_once_with("/tmp/x.png")


def test_ocr_loads_easyocr_on_demand():
    """Vision.ocr() lazily loads easyocr when it isn't loaded yet."""
    vision = _make_vision()
    easyocr = _easyocr_mock()

    def fake_load(load_moondream=True, load_easyocr=True):
        vision.easyocr = easyocr

    with mock.patch.object(vision, "load", side_effect=fake_load) as load:
        result = vision.ocr(path="img.png")

    load.assert_called_once_with(load_moondream=False)
    assert result == "hello world"


def test_ocr_returns_empty_and_prints_install_hint_on_import_error(capsys):
    """Vision.ocr() returns '' and prints the local-install hint when easyocr
    can't be imported."""
    vision = _make_vision()
    with mock.patch.object(vision, "load", side_effect=ImportError):
        result = vision.ocr(path="img.png")

    assert result == ""
    assert "pip install 'open-interpreter[local]'" in capsys.readouterr().out


def test_load_loads_easyocr_reader():
    """Vision.load(load_moondream=False) instantiates the easyocr Reader once."""
    vision = _make_vision()
    easyocr_module = mock.Mock()
    easyocr_module.Reader.return_value = "reader"
    with mock.patch.dict("sys.modules", {"easyocr": easyocr_module}):
        vision.load(load_moondream=False)

    easyocr_module.Reader.assert_called_once_with(["en"])
    assert vision.easyocr == "reader"


def test_load_loads_moondream_model(monkeypatch):
    """Vision.load() loads the moondream2 transformers model and returns True."""
    vision = _make_vision()
    transformers = mock.Mock()
    transformers.AutoModelForCausalLM.from_pretrained.return_value = "model"
    transformers.AutoTokenizer.from_pretrained.return_value = "tokenizer"
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "unset")
    with mock.patch.dict("sys.modules", {"transformers": transformers}):
        result = vision.load(load_easyocr=False)

    assert result is True
    assert vision.model == "model"
    assert vision.tokenizer == "tokenizer"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_query_uses_moondream_on_pil_image():
    """Vision.query(pil_image=...) encodes the image and asks moondream."""
    vision = _make_vision()
    model = mock.Mock()
    model.encode_image.return_value = "enc"
    model.answer_question.return_value = "answer"
    vision.model = model
    vision.tokenizer = mock.Mock()
    pil_image = Image.new("RGB", (4, 4))

    result = vision.query("What is this?", pil_image=pil_image)

    assert result == "answer"
    model.encode_image.assert_called_once_with(pil_image)
    model.answer_question.assert_called_once_with(
        "enc", "What is this?", vision.tokenizer, max_length=400
    )


def test_query_decodes_base64_image():
    """Vision.query(base_64=...) decodes the image before asking moondream."""
    vision = _make_vision()
    model = mock.Mock()
    model.encode_image.return_value = "enc"
    model.answer_question.return_value = "answer"
    vision.model = model
    vision.tokenizer = mock.Mock()

    result = vision.query(base_64=_png_base64())

    assert result == "answer"
    assert model.encode_image.call_args[0][0].size == (4, 4)


def test_query_loads_model_on_demand():
    """Vision.query() lazily loads the moondream model when not loaded."""
    vision = _make_vision()
    model = mock.Mock()
    model.encode_image.return_value = "enc"
    model.answer_question.return_value = "answer"

    def fake_load(load_moondream=True, load_easyocr=True):
        vision.model = model
        vision.tokenizer = mock.Mock()
        return True

    with mock.patch.object(vision, "load", side_effect=fake_load) as load:
        result = vision.query("What is this?", pil_image=Image.new("RGB", (4, 4)))

    load.assert_called_once_with(load_easyocr=False)
    assert result == "answer"


def test_query_returns_empty_on_import_error():
    """Vision.query() returns '' when the transformers import fails."""
    vision = _make_vision()
    with mock.patch.object(vision, "load", side_effect=ImportError):
        assert vision.query(pil_image=Image.new("RGB", (4, 4))) == ""


def test_query_returns_empty_when_load_fails():
    """Vision.query() returns '' when the model fails to load."""
    vision = _make_vision()
    with mock.patch.object(vision, "load", return_value=False):
        assert vision.query(pil_image=Image.new("RGB", (4, 4))) == ""
