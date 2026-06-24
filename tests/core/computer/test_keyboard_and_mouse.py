from types import SimpleNamespace
from unittest import mock


def test_keyboard_write_short_text_uses_clipboard():
    from interpreter.core.computer.keyboard.keyboard import Keyboard

    clipboard = SimpleNamespace(
        view=mock.Mock(return_value="history"),
        copy=mock.Mock(),
        paste=mock.Mock(),
    )
    computer = SimpleNamespace(clipboard=clipboard)
    keyboard = Keyboard(computer)

    fake_pyautogui = mock.Mock()
    with mock.patch("interpreter.core.computer.keyboard.keyboard.pyautogui", fake_pyautogui):
        with mock.patch("interpreter.core.computer.keyboard.keyboard.time.sleep"):
            keyboard.write("hi")

    clipboard.copy.assert_any_call("hi")
    clipboard.paste.assert_called()


def test_keyboard_press_delegates_to_pyautogui():
    from interpreter.core.computer.keyboard.keyboard import Keyboard

    computer = SimpleNamespace(clipboard=SimpleNamespace())
    keyboard = Keyboard(computer)
    fake_pyautogui = mock.Mock()

    with mock.patch("interpreter.core.computer.keyboard.keyboard.pyautogui", fake_pyautogui):
        with mock.patch("interpreter.core.computer.keyboard.keyboard.time.sleep"):
            keyboard.press("enter")

    fake_pyautogui.press.assert_called_once()


def test_mouse_scroll_calls_pyautogui():
    from interpreter.core.computer.mouse.mouse import Mouse

    computer = SimpleNamespace(
        display=SimpleNamespace(),
        emit_images=False,
        verbose=False,
    )
    mouse = Mouse(computer)
    fake_pyautogui = mock.Mock()

    with mock.patch("interpreter.core.computer.mouse.mouse.pyautogui", fake_pyautogui):
        mouse.scroll(3)

    fake_pyautogui.scroll.assert_called_once_with(3)


def test_mouse_move_rejects_too_many_positional_args():
    from interpreter.core.computer.mouse.mouse import Mouse

    mouse = Mouse(SimpleNamespace(display=SimpleNamespace(), emit_images=False))
    try:
        mouse.move(1, 2)
    except ValueError as e:
        assert "Too many positional arguments" in str(e)
    else:
        raise AssertionError("Expected ValueError")
