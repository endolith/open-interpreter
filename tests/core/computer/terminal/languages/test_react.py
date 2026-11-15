from unittest import mock

from interpreter.core.computer.terminal.languages.react import (
    React,
    is_incompatible,
    template,
)


def test_is_incompatible_detects_require():
    """is_incompatible() flags CommonJS require() calls in the first lines."""
    assert is_incompatible("require('react')") is True


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


def test_react_run_omits_image_when_the_browser_cannot_render():
    """React.run() yields no image chunk when rendering returns None.

    html_to_png_base64() reports a missing or failing headless browser by
    returning None. An image chunk whose content is None would hand the
    assistant a broken attachment, so the chunk is omitted and the console
    line above still tells it React is on screen.
    """
    react = React()
    code = "function App(){return <div/>}"
    with mock.patch(
        "interpreter.core.computer.terminal.languages.react.html_to_png_base64",
        return_value=None,
    ) as to_png:
        chunks = list(react.run(code))

    assert chunks == [
        {
            "type": "console",
            "format": "output",
            "content": "React is being displayed on the user's machine...",
            "recipient": "assistant",
        },
        {
            "type": "code",
            "format": "html",
            "content": template.replace("{insert_react_code}", code),
            "recipient": "user",
        },
    ]
    to_png.assert_called_once_with(template.replace("{insert_react_code}", code))
