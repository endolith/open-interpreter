from unittest import mock

from interpreter.core.computer.terminal.languages.react import React, is_incompatible


def test_is_incompatible_detects_import():
    assert is_incompatible("import React from 'react'\nexport default App")


def test_is_incompatible_allows_simple_jsx():
    assert is_incompatible("function App() { return <div />; }") is False


def test_react_incompatible_yields_error():
    react = React()
    chunks = list(react.run("import X from 'x'"))
    assert chunks[0]["type"] == "console"
    assert "not supported" in chunks[0]["content"].lower()


def test_react_compatible_yields_image_chunk():
    react = React()
    with mock.patch(
        "interpreter.core.computer.terminal.languages.react.html_to_png_base64",
        return_value="imgdata",
    ):
        chunks = list(react.run("function App(){return <div/>}"))
    image_chunks = [c for c in chunks if c["type"] == "image"]
    assert len(image_chunks) == 1
    assert image_chunks[0]["format"] == "base64.png"
    assert image_chunks[0]["content"] == "imgdata"
