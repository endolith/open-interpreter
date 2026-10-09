from unittest import mock

from interpreter.terminal_interface.utils.display_output import (
    display_output,
    display_output_cli,
    open_file,
)


def test_display_output_cli_console(capsys):
    """Console output is printed directly to stdout."""
    display_output_cli({"type": "console", "content": "hello"})
    assert capsys.readouterr().out.strip() == "hello"


def test_display_output_cli_html_writes_temp_file():
    """HTML code output is written to a temp file and opened in the browser."""
    output = {"type": "code", "format": "html", "content": "<html></html>"}
    with mock.patch(
        "interpreter.terminal_interface.utils.display_output.open_file"
    ) as open_file_mock:
        display_output_cli(output)
    open_file_mock.assert_called_once()


def test_display_output_jupyter_delegates():
    """In a Jupyter notebook, display_output skips the CLI path and returns a status string."""
    with mock.patch(
        "interpreter.terminal_interface.utils.display_output.in_jupyter_notebook",
        return_value=True,
    ):
        with mock.patch(
            "interpreter.terminal_interface.utils.display_output.display_output_cli"
        ) as cli:
            result = display_output({"type": "console", "content": "x"})
    cli.assert_not_called()
    assert result == "Displayed on the user's machine."


def test_open_file_linux():
    """open_file on Linux invokes xdg-open with the file path."""
    with mock.patch("platform.system", return_value="Linux"):
        with mock.patch("subprocess.run") as run:
            open_file("/tmp/x.html")
    run.assert_called_once_with(["xdg-open", "/tmp/x.html"])


def test_open_file_darwin():
    """open_file on macOS invokes open with the file path."""
    with mock.patch("platform.system", return_value="Darwin"):
        with mock.patch("subprocess.run") as run:
            open_file("/tmp/x.html")
    run.assert_called_once_with(["open", "/tmp/x.html"])


def test_open_file_windows():
    """open_file on Windows uses os.startfile to open the file."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch(
            "interpreter.terminal_interface.utils.display_output.os.startfile",
            create=True,
        ) as startfile:
            open_file("C:\\x.html")
    startfile.assert_called_once_with("C:\\x.html")


def test_display_output_cli_base64_image_writes_decoded_temp_file(monkeypatch):
    """A base64 image is decoded and written to a temp file with the format's extension.

    The extension comes from the part of "base64.png" after the dot; a dropped
    extension or an undecoded payload would produce a corrupt/unopenable file.
    The path is captured at open time and the bytes are read after the call,
    once the file has been closed and flushed.
    """
    import base64
    import os

    opened = {}

    monkeypatch.setattr(
        "interpreter.terminal_interface.utils.display_output.open_file",
        lambda path: opened.setdefault("path", path),
    )

    payload = b"\x89PNG-bytes"
    display_output_cli(
        {
            "type": "image",
            "format": "base64.png",
            "content": base64.b64encode(payload).decode(),
        }
    )

    assert opened["path"].endswith(".png")
    with open(opened["path"], "rb") as fh:
        assert fh.read() == payload
    os.remove(opened["path"])


def test_display_output_cli_base64_without_dot_defaults_to_png(monkeypatch):
    """A base64 image format with no dot defaults the suffix to ".png".

    The format may be just "base64" (no extension); the fallback extension must
    be "png" so the temp file is recoverable.
    """
    import base64
    import os

    opened = {}

    def capture_open(path):
        opened["path"] = path

    monkeypatch.setattr(
        "interpreter.terminal_interface.utils.display_output.open_file",
        capture_open,
    )

    try:
        display_output_cli(
            {"type": "image", "format": "base64", "content": base64.b64encode(b"x").decode()}
        )

        assert opened["path"].endswith(".png")
    finally:
        # Removed here rather than inside capture_open: with the file flushed
        # and closed before open_file runs, removal here is always safe.
        if "path" in opened and os.path.exists(opened["path"]):
            os.remove(opened["path"])


def test_display_output_cli_path_image_opens_the_file_directly(monkeypatch):
    """An image given as a filesystem path is opened without a temp file.

    The "path" branch must open the given file, not decode its content as
    base64 (the format check is what separates them).
    """
    opened = {}
    monkeypatch.setattr(
        "interpreter.terminal_interface.utils.display_output.open_file",
        lambda path: opened.setdefault("path", path),
    )

    display_output_cli(
        {"type": "image", "format": "path", "content": "/tmp/picture.png"}
    )

    assert opened["path"] == "/tmp/picture.png"


def test_display_output_cli_javascript_writes_js_temp_file(monkeypatch):
    """JavaScript output is written to a ".js" temp file and opened.

    The suffix distinguishes it from the html branch. The path is captured at
    open time and the content read after the call (the write is still buffered
    while the file is open), pinning current behavior.
    """
    import os

    opened = {}

    monkeypatch.setattr(
        "interpreter.terminal_interface.utils.display_output.open_file",
        lambda path: opened.setdefault("path", path),
    )

    display_output_cli(
        {"type": "code", "format": "javascript", "content": "console.log(1)"}
    )

    assert opened["path"].endswith(".js")
    with open(opened["path"]) as fh:
        assert fh.read() == "console.log(1)"
    os.remove(opened["path"])


def test_display_output_cli_opens_flushed_content(monkeypatch):
    """open_file sees the written bytes at call time, not an empty file (issue #372).

    open_file used to run inside the NamedTemporaryFile block while the write
    was still buffered, so the OS viewer opened an empty file. Reading the
    path inside the open_file stand-in fails if the ordering regresses.
    """
    import os

    seen = {}

    def capture_open(path):
        with open(path, "rb") as fh:
            seen["content"] = fh.read()
        seen["path"] = path

    monkeypatch.setattr(
        "interpreter.terminal_interface.utils.display_output.open_file",
        capture_open,
    )

    cases = [
        ({"type": "code", "format": "html", "content": "<html></html>"}, b"<html></html>"),
        ({"type": "code", "format": "javascript", "content": "console.log(1)"}, b"console.log(1)"),
    ]
    for output, expected in cases:
        seen.clear()
        display_output_cli(output)
        assert seen["content"] == expected
        os.remove(seen["path"])


def test_display_output_jupyter_console_prints(monkeypatch, capsys):
    """In Jupyter, a console chunk is printed, not routed to the CLI path.

    The notebook branch handles console itself so the output lands in the cell;
    if the type check were wrong, the chunk would either vanish or take the
    image branch.
    """
    display_module = "interpreter.terminal_interface.utils.display_output"
    monkeypatch.setattr(f"{display_module}.in_jupyter_notebook", lambda: True)
    monkeypatch.setattr(f"{display_module}.display_output_cli", mock.Mock())

    with mock.patch("IPython.display.display"), mock.patch(
        "IPython.display.HTML"
    ), mock.patch("IPython.display.Image"), mock.patch(
        "IPython.display.Javascript"
    ):
        result = display_output({"type": "console", "content": "in-notebook"})

    assert result == "Displayed on the user's machine."
    assert "in-notebook" in capsys.readouterr().out


def test_display_output_jupyter_image_base64_decodes_and_displays(monkeypatch):
    """In Jupyter, a base64 image chunk is decoded and passed to IPython.display.Image.

    The decode is what turns the payload into displayable bytes; a missing
    decode or a wrong format check would hand Image the raw base64 string.
    """
    import base64

    display_module = "interpreter.terminal_interface.utils.display_output"
    monkeypatch.setattr(f"{display_module}.in_jupyter_notebook", lambda: True)
    monkeypatch.setattr(f"{display_module}.display_output_cli", mock.Mock())

    seen = {}
    with mock.patch("IPython.display.display") as display, mock.patch(
        "IPython.display.Image"
    ) as image, mock.patch("IPython.display.HTML"), mock.patch(
        "IPython.display.Javascript"
    ):
        image.side_effect = lambda *a, **k: seen.setdefault("args", a)
        payload = b"img-bytes"
        display_output(
            {
                "type": "image",
                "format": "base64.png",
                "content": base64.b64encode(payload).decode(),
            }
        )

    assert seen["args"] == (payload,)
    display.assert_called_once()


def test_display_output_jupyter_html_and_javascript(monkeypatch):
    """In Jupyter, html and javascript chunks are wrapped in IPython HTML/Javascript.

    Flattening the two branches to one display call would lose the mime type the
    notebook needs to render each correctly.
    """
    display_module = "interpreter.terminal_interface.utils.display_output"
    monkeypatch.setattr(f"{display_module}.in_jupyter_notebook", lambda: True)
    monkeypatch.setattr(f"{display_module}.display_output_cli", mock.Mock())

    with mock.patch("IPython.display.display") as display, mock.patch(
        "IPython.display.HTML"
    ) as html, mock.patch("IPython.display.Image"), mock.patch(
        "IPython.display.Javascript"
    ) as js:
        display_output({"type": "code", "format": "html", "content": "<b>x</b>"})
        display_output({"type": "code", "format": "javascript", "content": "1"})

    html.assert_called_once_with("<b>x</b>")
    js.assert_called_once_with("1")
    # Assert the wrapped objects and their order, not just how many times
    # display was called: a count passes even if display received the wrong
    # objects, or the javascript wrapped before the html.
    display.assert_has_calls(
        [mock.call(html.return_value), mock.call(js.return_value)]
    )
