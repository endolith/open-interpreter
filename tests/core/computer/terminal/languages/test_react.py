from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.terminal.languages.react import React, is_incompatible


def _computer(supports_vision):
    """The slice of Computer that decides whether the assistant sees a PNG."""
    return SimpleNamespace(
        interpreter=SimpleNamespace(
            llm=SimpleNamespace(supports_vision=supports_vision)
        )
    )


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
    react = React(_computer(supports_vision=True))
    with mock.patch(
        "interpreter.core.computer.terminal.languages.react.html_to_png_base64",
        return_value="imgdata",
    ):
        chunks = list(react.run("function App(){return <div/>}"))
    image_chunks = [c for c in chunks if c["type"] == "image"]
    assert len(image_chunks) == 1
    assert image_chunks[0]["format"] == "base64.png"
    assert image_chunks[0]["content"] == "imgdata"
    assert chunks[0]["type"] == "code"


def test_react_non_vision_yields_one_source_console():
    """A non-vision model gets one assistant result: the source, never a PNG.

    The old sequence was narration plus a screenshot regardless of the model,
    so one tool call arrived as several results (#553).
    """
    react = React(_computer(supports_vision=False))
    code = "function App(){return <div/>}"
    with mock.patch(
        "interpreter.core.computer.terminal.languages.react.html_to_png_base64",
        return_value="imgdata",
    ) as to_png:
        chunks = list(react.run(code))

    assistant_results = [c for c in chunks if c.get("recipient") == "assistant"]
    assert len(assistant_results) == 1
    assert code in assistant_results[0]["content"]
    assert all(c["type"] != "image" for c in chunks)
    to_png.assert_not_called()


def test_react_falls_back_to_one_console_when_the_screenshot_fails():
    """A failed screenshot must not raise, and must not add a second result."""
    react = React(_computer(supports_vision=True))
    with mock.patch(
        "interpreter.core.computer.terminal.languages.react.html_to_png_base64",
        return_value=None,
    ):
        chunks = list(react.run("function App(){return <div/>}"))

    assistant_results = [c for c in chunks if c.get("recipient") == "assistant"]
    assert len(assistant_results) == 1
    assert "headless browser failed" in assistant_results[0]["content"]
    assert "function App()" in assistant_results[0]["content"]
    assert all(c["type"] != "image" for c in chunks)


def test_react_without_a_computer_yields_the_source():
    """Constructed bare, React never reaches for the browser."""
    with mock.patch(
        "interpreter.core.computer.terminal.languages.react.html_to_png_base64",
        return_value="imgdata",
    ) as to_png:
        chunks = list(React().run("function App(){return <div/>}"))

    assert all(c["type"] != "image" for c in chunks)
    to_png.assert_not_called()
