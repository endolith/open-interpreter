from types import SimpleNamespace
from unittest import mock

from interpreter.core.terminal.languages.react import React, is_incompatible


def _vision_interpreter():
    return SimpleNamespace(llm=SimpleNamespace(supports_vision=True))


def test_is_incompatible_detects_import():
    """is_incompatible() flags React code that uses ES module import syntax."""
    assert is_incompatible("import React from 'react'\nexport default App")


def test_is_incompatible_allows_simple_jsx():
    """is_incompatible() allows self-contained JSX without import statements."""
    assert is_incompatible("function App() { return <div />; }") is False


def test_react_incompatible_yields_error():
    """React.run() yields a console error when the source uses unsupported imports."""
    react = React()
    chunks = list(react.run("import X from 'x'"))
    assert chunks[0]["type"] == "console"
    assert "not supported" in chunks[0]["content"].lower()


def test_react_compatible_yields_image_chunk():
    """React.run() renders compatible JSX to a base64 PNG image chunk."""
    react = React(_vision_interpreter())
    with mock.patch(
        "interpreter.core.terminal.languages.react.html_to_png_base64",
        return_value="imgdata",
    ):
        chunks = list(react.run("function App(){return <div/>}"))
    image_chunks = [c for c in chunks if c["type"] == "image"]
    assert len(image_chunks) == 1
    assert image_chunks[0]["format"] == "base64.png"
    assert image_chunks[0]["content"] == "imgdata"


def test_react_non_vision_yields_source_console_chunk():
    """Non-vision models get the rendered HTML source instead of a screenshot."""
    react = React(SimpleNamespace(llm=SimpleNamespace(supports_vision=False)))
    with mock.patch(
        "interpreter.core.terminal.languages.react.html_to_png_base64",
        return_value="imgdata",
    ) as to_png:
        chunks = list(react.run("function App(){return <div/>}"))
    assert any(c["type"] == "console" for c in chunks)
    to_png.assert_not_called()


def test_react_run_falls_back_to_source_when_screenshot_fails():
    """A failed React screenshot shows the source instead of raising into respond()."""
    react = React(_vision_interpreter())
    with mock.patch(
        "interpreter.core.terminal.languages.react.html_to_png_base64",
        return_value=None,
    ):
        chunks = list(react.run("function App(){return <div/>}"))

    assert all(chunk["type"] != "image" for chunk in chunks)
    assert any("headless browser failed" in c.get("content", "") for c in chunks)
    assert any("function App(){return <div/>}" in c.get("content", "") for c in chunks)
