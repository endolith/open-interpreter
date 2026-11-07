from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.terminal.languages.html import HTML


def _computer(supports_vision):
    """The slice of Computer that decides whether the assistant sees a PNG."""
    return SimpleNamespace(
        interpreter=SimpleNamespace(
            llm=SimpleNamespace(supports_vision=supports_vision)
        )
    )


def test_html_run_yields_code_then_image():
    """For a vision model the chunks are the code chunk and the PNG, in order.

    The image is the assistant's one result; a narration console alongside it
    would be a second assistant-facing message for a single tool call.
    """
    html = HTML(_computer(supports_vision=True))
    with mock.patch(
        "interpreter.core.computer.terminal.languages.html.html_to_png_base64",
        return_value="base64data",
    ):
        chunks = list(html.run("<html><body>Hi</body></html>"))

    assert chunks[0]["type"] == "code"
    assert chunks[1]["type"] == "image"
    assert chunks[1]["format"] == "base64.png"
    assert chunks[1]["content"] == "base64data"
    assert [c for c in chunks if c.get("recipient") == "assistant"] == [chunks[1]]


def test_html_run_chunk_dicts_are_exact():
    """Each yielded chunk equals its full expected dict.

    The chunks are the language's public protocol: the assistant/user routing
    (recipient), the format keys, and the code chunk passing the source
    through unchanged are all read by the message pipeline. Asserting only
    `type` let every key rename and value change survive, so the whole dicts
    are pinned.
    """
    html = HTML(_computer(supports_vision=True))
    code = "<html><body>Hi</body></html>"
    with mock.patch(
        "interpreter.core.computer.terminal.languages.html.html_to_png_base64",
        return_value="base64data",
    ) as to_png:
        chunks = list(html.run(code))

    assert chunks == [
        {"type": "code", "format": "html", "content": code, "recipient": "user"},
        {
            "type": "image",
            "format": "base64.png",
            "content": "base64data",
            "recipient": "assistant",
        },
    ]
    to_png.assert_called_once_with(code)


def test_html_run_non_vision_yields_one_source_console():
    """A non-vision model gets one assistant result: the source, never a PNG.

    The old sequence was narration plus a screenshot regardless of the model,
    so one tool call arrived as several results and the model lost track of
    which output belonged to which call.
    """
    html = HTML(_computer(supports_vision=False))
    code = "<html><body>Hi</body></html>"
    with mock.patch(
        "interpreter.core.computer.terminal.languages.html.html_to_png_base64",
        return_value="base64data",
    ) as to_png:
        chunks = list(html.run(code))

    assistant_results = [c for c in chunks if c.get("recipient") == "assistant"]
    assert len(assistant_results) == 1
    assert code in assistant_results[0]["content"]
    assert all(c["type"] != "image" for c in chunks)
    to_png.assert_not_called()


def test_html_run_falls_back_to_one_console_when_the_screenshot_fails():
    """A failed screenshot must not raise, and must not add a second result."""
    html = HTML(_computer(supports_vision=True))
    with mock.patch(
        "interpreter.core.computer.terminal.languages.html.html_to_png_base64",
        return_value=None,
    ):
        chunks = list(html.run("<p>hi</p>"))

    assistant_results = [c for c in chunks if c.get("recipient") == "assistant"]
    assert len(assistant_results) == 1
    assert "headless browser failed" in assistant_results[0]["content"]
    assert "<p>hi</p>" in assistant_results[0]["content"]
    assert all(c["type"] != "image" for c in chunks)


def test_html_run_without_a_computer_yields_the_source():
    """Constructed bare (as tests and custom integrations do), no PNG is taken."""
    html = HTML()
    with mock.patch(
        "interpreter.core.computer.terminal.languages.html.html_to_png_base64",
        return_value="base64data",
    ) as to_png:
        chunks = list(html.run("<p>hi</p>"))

    assert all(c["type"] != "image" for c in chunks)
    to_png.assert_not_called()


def test_html_run_falls_back_to_source_when_the_screenshot_raises():
    """A renderer that raises is a soft failure, not a failed run.

    `html_to_png_base64` opens the screenshot file it just wrote; a missing file
    raises `FileNotFoundError`. That must produce the source fallback rather
    than escaping `run`.
    """
    html = HTML(_computer(supports_vision=True))
    with mock.patch(
        "interpreter.core.computer.terminal.languages.html.html_to_png_base64",
        side_effect=FileNotFoundError("screenshot vanished"),
    ):
        chunks = list(html.run("<p>hi</p>"))

    assistant_results = [c for c in chunks if c.get("recipient") == "assistant"]
    assert len(assistant_results) == 1
    assert "headless browser failed" in assistant_results[0]["content"]
    assert "<p>hi</p>" in assistant_results[0]["content"]
    assert all(c["type"] != "image" for c in chunks)
