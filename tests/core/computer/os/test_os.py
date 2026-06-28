from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.os.os import Os


def test_get_selected_text_uses_clipboard():
    """get_selected_text copies selection via clipboard and restores prior contents."""
    clipboard = SimpleNamespace(
        view=mock.Mock(side_effect=["previous", "selected text"]),
        copy=mock.Mock(),
    )
    os_module = Os(SimpleNamespace(clipboard=clipboard))
    assert os_module.get_selected_text() == "selected text"
    clipboard.copy.assert_any_call()
    clipboard.copy.assert_any_call("previous")
    assert clipboard.copy.call_count == 2


def test_notify_truncates_long_text_on_linux():
    """Notifications longer than 200 chars are shortened before plyer.notify."""
    import sys

    os_module = Os(SimpleNamespace(verbose=False))
    plyer = mock.Mock()
    with mock.patch("interpreter.core.computer.os.os.platform.system", return_value="Linux"):
        with mock.patch.dict(sys.modules, {"plyer": plyer}):
            os_module.notify("x" * 300)
    plyer.notification.notify.assert_called_once()
    message = plyer.notification.notify.call_args.kwargs["message"]
    assert len(message) <= 203  # 200 chars + "..."
    assert message.endswith("...")


def test_notify_macos_runs_osascript():
    os_module = Os(SimpleNamespace(verbose=False))
    with mock.patch(
        "interpreter.core.computer.os.os.platform.system", return_value="Darwin"
    ):
        with mock.patch(
            "interpreter.core.computer.os.os.subprocess.run"
        ) as run:
            os_module.notify('Say "hello"')
    run.assert_called_once_with(["osascript", "-e", mock.ANY])
    script = run.call_args[0][0][2]
    assert "display notification" in script
