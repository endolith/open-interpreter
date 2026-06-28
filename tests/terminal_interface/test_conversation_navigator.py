import json
from types import SimpleNamespace
from unittest import mock

from interpreter.terminal_interface.conversation_navigator import (
    conversation_navigator,
    open_folder,
)


def test_conversation_navigator_missing_dir(tmp_path, capsys):
    interpreter = SimpleNamespace(
        display_message=mock.Mock(),
        messages=[],
        chat=mock.Mock(),
    )
    with mock.patch(
        "interpreter.terminal_interface.conversation_navigator.get_storage_path",
        return_value=str(tmp_path / "missing"),
    ):
        conversation_navigator(interpreter)
    assert "No conversations found" in capsys.readouterr().out


def test_conversation_navigator_loads_selected_conversation(tmp_path):
    conv_dir = tmp_path / "conversations"
    conv_dir.mkdir()
    messages = [{"role": "user", "type": "message", "content": "hi"}]
    (conv_dir / "test_chat__2024.json").write_text(json.dumps(messages))

    interpreter = SimpleNamespace(
        display_message=mock.Mock(),
        messages=[],
        chat=mock.Mock(),
        conversation_filename=None,
    )

    with mock.patch(
        "interpreter.terminal_interface.conversation_navigator.get_storage_path",
        return_value=str(conv_dir),
    ):
        with mock.patch(
            "interpreter.terminal_interface.conversation_navigator.inquirer.prompt",
            return_value={"name": "test chat... (2024)"},
        ):
            with mock.patch(
                "interpreter.terminal_interface.conversation_navigator.render_past_conversation"
            ):
                conversation_navigator(interpreter)

    assert interpreter.messages == messages
    interpreter.chat.assert_called_once()


def test_open_folder_linux():
    with mock.patch("platform.system", return_value="Linux"):
        with mock.patch("subprocess.run") as run:
            open_folder("/tmp/test")
    run.assert_called_once_with(["xdg-open", "/tmp/test"])


def test_open_folder_darwin():
    with mock.patch("platform.system", return_value="Darwin"):
        with mock.patch("subprocess.run") as run:
            open_folder("/tmp/test")
    run.assert_called_once_with(["open", "/tmp/test"])


def test_open_folder_windows():
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch(
            "interpreter.terminal_interface.conversation_navigator.os.startfile",
            create=True,
        ) as startfile:
            open_folder("C:\\conversations")
    startfile.assert_called_once_with("C:\\conversations")
