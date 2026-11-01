from types import SimpleNamespace
from unittest import mock

from interpreter.core.terminal.languages.html import HTML


def _vision_interpreter():
    return SimpleNamespace(llm=SimpleNamespace(supports_vision=True))


def test_html_run_yields_console_code_and_image():
    """HTML.run() yields console, code, and base64 PNG image chunks in order."""
    html = HTML(_vision_interpreter())
    with mock.patch(
        "interpreter.core.terminal.languages.html.html_to_png_base64",
        return_value="base64data",
    ):
        chunks = list(html.run("<html><body>Hi</body></html>"))

    assert chunks[0]["type"] == "code"
    assert chunks[1]["type"] == "console"
    assert chunks[2]["type"] == "image"
    assert chunks[2]["format"] == "base64.png"
    assert chunks[2]["content"] == "base64data"


def test_html_run_chunk_dicts_are_exact():
    """Each yielded chunk equals its full expected dict.

    The chunks are the language's public protocol: the assistant/user routing
    (recipient), the format keys, and the second chunk passing the code through
    unchanged are all read by the message pipeline. Asserting only `type` let
    every key rename and value change survive, so the whole dicts are pinned.
    """
    html = HTML(_vision_interpreter())
    code = "<html><body>Hi</body></html>"
    with mock.patch(
        "interpreter.core.terminal.languages.html.html_to_png_base64",
        return_value="base64data",
    ) as to_png:
        chunks = list(html.run(code))

    assert chunks == [
        {"type": "code", "format": "html", "content": code, "recipient": "user"},
        {
            "type": "console",
            "format": "output",
            "content": "HTML is being displayed on the user's machine...",
            "recipient": "assistant",
        },
        {
            "type": "image",
            "format": "base64.png",
            "content": "base64data",
            "recipient": "assistant",
        },
    ]
    to_png.assert_called_once_with(code)



def test_html_run_non_vision_yields_source_console_chunk():
    """Non-vision models get the HTML source described in a console chunk, not a PNG."""
    html = HTML(SimpleNamespace(llm=SimpleNamespace(supports_vision=False)))
    code = "<html><body>Hi</body></html>"
    with mock.patch(
        "interpreter.core.terminal.languages.html.html_to_png_base64",
        return_value="base64data",
    ) as to_png:
        chunks = list(html.run(code))
    assert chunks[1]["type"] == "console"
    assert code in chunks[1]["content"]
    to_png.assert_not_called()


def test_html_run_falls_back_to_source_when_screenshot_fails():
    """A failed screenshot must not raise into the run loop; the source is shown."""
    html = HTML(_vision_interpreter())
    with mock.patch(
        "interpreter.core.terminal.languages.html.html_to_png_base64",
        return_value=None,
    ):
        chunks = list(html.run("<p>hi</p>"))

    assert all(chunk["type"] != "image" for chunk in chunks)
    assert any("headless browser failed" in c.get("content", "") for c in chunks)
    assert any("<p>hi</p>" in c.get("content", "") for c in chunks)
