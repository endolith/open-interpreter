from unittest import mock

from interpreter.core.computer.terminal.languages.html import HTML


def test_html_run_yields_console_code_and_image():
    """HTML.run() yields console, code, and base64 PNG image chunks in order."""
    html = HTML()
    with mock.patch(
        "interpreter.core.computer.terminal.languages.html.html_to_png_base64",
        return_value="base64data",
    ):
        chunks = list(html.run("<html><body>Hi</body></html>"))

    assert chunks[0]["type"] == "console"
    assert chunks[1]["type"] == "code"
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
    html = HTML()
    code = "<html><body>Hi</body></html>"
    with mock.patch(
        "interpreter.core.computer.terminal.languages.html.html_to_png_base64",
        return_value="base64data",
    ) as to_png:
        chunks = list(html.run(code))

    assert chunks == [
        {
            "type": "console",
            "format": "output",
            "content": "HTML being displayed on the user's machine...",
            "recipient": "assistant",
        },
        {"type": "code", "format": "html", "content": code, "recipient": "user"},
        {
            "type": "image",
            "format": "base64.png",
            "content": "base64data",
            "recipient": "assistant",
        },
    ]
    to_png.assert_called_once_with(code)

