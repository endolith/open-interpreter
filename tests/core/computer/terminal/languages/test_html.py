from unittest import mock

from interpreter.core.computer.terminal.languages.html import HTML


def test_html_run_yields_console_code_and_image():
    html = HTML()
    with mock.patch(
        "interpreter.core.computer.terminal.languages.html.html_to_png_base64",
        return_value="base64data",
    ):
        chunks = list(html.run("<html><body>Hi</body></html>"))

    assert chunks[0]["type"] == "console"
    assert chunks[1]["type"] == "code"
    assert chunks[2]["type"] == "image"
