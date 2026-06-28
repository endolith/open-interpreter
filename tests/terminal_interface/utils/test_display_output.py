from unittest import mock

from interpreter.terminal_interface.utils.display_output import (
    display_output,
    display_output_cli,
    open_file,
)


def test_display_output_cli_console(capsys):
    display_output_cli({"type": "console", "content": "hello"})
    assert capsys.readouterr().out.strip() == "hello"


def test_display_output_cli_html_writes_temp_file():
    output = {"type": "code", "format": "html", "content": "<html></html>"}
    with mock.patch(
        "interpreter.terminal_interface.utils.display_output.open_file"
    ) as open_file_mock:
        display_output_cli(output)
    open_file_mock.assert_called_once()


def test_display_output_jupyter_delegates():
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
    with mock.patch("platform.system", return_value="Linux"):
        with mock.patch("subprocess.run") as run:
            open_file("/tmp/x.html")
    run.assert_called_once_with(["xdg-open", "/tmp/x.html"])


def test_open_file_darwin():
    with mock.patch("platform.system", return_value="Darwin"):
        with mock.patch("subprocess.run") as run:
            open_file("/tmp/x.html")
    run.assert_called_once_with(["open", "/tmp/x.html"])


def test_open_file_windows():
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch(
            "interpreter.terminal_interface.utils.display_output.os.startfile",
            create=True,
        ) as startfile:
            open_file("C:\\x.html")
    startfile.assert_called_once_with("C:\\x.html")
