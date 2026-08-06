"""Characterization tests for ``computer.keyboard``.

These mock pyautogui and the clipboard so they run headless in CI. They trip
when the ``computer/* -> toolbox/*`` port renames or reworks the keyboard
subsystem.
"""

from types import SimpleNamespace
from unittest import mock

import pytest


def _make_keyboard(computer=None):
    from interpreter.core.computer.keyboard.keyboard import Keyboard

    if computer is None:
        computer = SimpleNamespace(clipboard=_make_clipboard())
    return Keyboard(computer)


def _make_clipboard(history="history"):
    return SimpleNamespace(
        view=mock.Mock(return_value=history),
        copy=mock.Mock(),
        paste=mock.Mock(),
    )


def _patch_keyboard_deps():
    pyautogui = mock.Mock()
    patcher = mock.patch(
        "interpreter.core.computer.keyboard.keyboard.pyautogui", pyautogui
    )
    sleep = mock.patch("interpreter.core.computer.keyboard.keyboard.time.sleep")
    return patcher, sleep, pyautogui


def test_write_short_text_uses_clipboard():
    """Keyboard.write() copies short text to the clipboard and pastes it."""
    clipboard = _make_clipboard()
    keyboard = _make_keyboard(SimpleNamespace(clipboard=clipboard))

    patcher, sleep, pyautogui = _patch_keyboard_deps()
    with patcher, sleep:
        keyboard.write("hi")

    clipboard.copy.assert_any_call("hi")
    clipboard.paste.assert_called()
    # The prior clipboard contents are restored afterwards.
    clipboard.copy.assert_any_call("history")
    pyautogui.write.assert_not_called()


def test_write_with_interval_types_via_pyautogui():
    """Keyboard.write(interval=) types through pyautogui.write instead of the
    clipboard."""
    clipboard = _make_clipboard()
    keyboard = _make_keyboard(SimpleNamespace(clipboard=clipboard))

    with mock.patch("interpreter.core.computer.keyboard.keyboard.pyautogui", mock.Mock()) as pyautogui:
        with mock.patch("interpreter.core.computer.keyboard.keyboard.time.sleep"):
            keyboard.write("hi", interval=0.05)

    pyautogui.write.assert_called_once_with("hi", interval=0.05)
    clipboard.copy.assert_not_called()
    clipboard.paste.assert_not_called()


def test_write_trailing_newline_presses_enter():
    """Keyboard.write() strips a trailing newline and presses Enter afterwards."""
    clipboard = _make_clipboard()
    keyboard = _make_keyboard(SimpleNamespace(clipboard=clipboard))

    patcher, sleep, pyautogui = _patch_keyboard_deps()
    with patcher, sleep:
        keyboard.write("hi\n")

    clipboard.copy.assert_any_call("hi")
    clipboard.paste.assert_called()
    pyautogui.press.assert_called_once_with(("enter",), presses=1, interval=0.1)


def test_write_multiline_short_text_pastes_line_by_line():
    """Keyboard.write() pastes each line separately when there are fewer than 5."""
    clipboard = _make_clipboard()
    keyboard = _make_keyboard(SimpleNamespace(clipboard=clipboard))

    _, sleep, _ = _patch_keyboard_deps()
    with mock.patch("interpreter.core.computer.keyboard.keyboard.pyautogui", mock.Mock()), sleep:
        keyboard.write("a\nb")

    assert [call.args for call in clipboard.copy.call_args_list] == [
        ("a\n",),
        ("b",),
        ("history",),
    ]
    assert clipboard.paste.call_count == 2


def test_write_long_text_pastes_once():
    """Keyboard.write() pastes the whole block at once for 5 or more lines."""
    clipboard = _make_clipboard()
    keyboard = _make_keyboard(SimpleNamespace(clipboard=clipboard))
    long_text = "a\nb\nc\nd\ne"

    _, sleep, _ = _patch_keyboard_deps()
    with mock.patch("interpreter.core.computer.keyboard.keyboard.pyautogui", mock.Mock()), sleep:
        keyboard.write(long_text)

    clipboard.copy.assert_any_call(long_text)
    assert clipboard.copy.call_count == 2  # once for the text, once to restore
    assert clipboard.paste.call_count == 1


def test_press_delegates_to_pyautogui():
    """Keyboard.press() forwards key presses to pyautogui with default timing."""
    keyboard = _make_keyboard()

    with mock.patch("interpreter.core.computer.keyboard.keyboard.pyautogui", mock.Mock()) as pyautogui:
        with mock.patch("interpreter.core.computer.keyboard.keyboard.time.sleep"):
            keyboard.press("enter")

    pyautogui.press.assert_called_once_with(("enter",), presses=1, interval=0.1)


def test_press_and_release_proxies_press():
    """Keyboard.press_and_release() is a perfect proxy for press()."""
    keyboard = _make_keyboard()

    with mock.patch.object(keyboard, "press", return_value=None) as press:
        result = keyboard.press_and_release("a", presses=2, interval=0.2)

    press.assert_called_once_with("a", presses=2, interval=0.2)
    assert result is None


def test_hotkey_non_darwin_uses_pyautogui():
    """Keyboard.hotkey() falls back to pyautogui.hotkey on non-macOS systems."""
    keyboard = _make_keyboard()

    with mock.patch(
        "interpreter.core.computer.keyboard.keyboard.platform.system",
        return_value="Linux",
    ):
        with mock.patch("interpreter.core.computer.keyboard.keyboard.pyautogui", mock.Mock()) as pyautogui:
            with mock.patch("interpreter.core.computer.keyboard.keyboard.time.sleep"):
                keyboard.hotkey("ctrl", "x")

    pyautogui.hotkey.assert_called_once_with("ctrl", "x", interval=0.1)


def _hotkey_on_darwin(keyboard, *args):
    with mock.patch(
        "interpreter.core.computer.keyboard.keyboard.platform.system",
        return_value="Darwin",
    ):
        with mock.patch("interpreter.core.computer.keyboard.keyboard.os.system", mock.Mock()) as os_system:
            with mock.patch("interpreter.core.computer.keyboard.keyboard.time.sleep"):
                keyboard.hotkey(*args)
    return os_system


def test_hotkey_darwin_uses_osascript():
    """Keyboard.hotkey() drives AppleScript through os.system on macOS."""
    keyboard = _make_keyboard()
    os_system = _hotkey_on_darwin(keyboard, "a", "command")
    assert os_system.call_count == 1
    assert 'keystroke "a" using command down' in os_system.call_args[0][0]


def test_hotkey_darwin_reorders_modifier_first():
    """Keyboard.hotkey() on macOS accepts the modifier as the first argument."""
    keyboard = _make_keyboard()
    os_system = _hotkey_on_darwin(keyboard, "command", "a")
    assert 'keystroke "a" using command down' in os_system.call_args[0][0]


def test_hotkey_darwin_maps_space_and_enter():
    """Keyboard.hotkey() maps "space" and "enter" keystrokes to AppleScript
    escape sequences on macOS."""
    keyboard = _make_keyboard()

    space = _hotkey_on_darwin(keyboard, "space", "command")
    assert 'keystroke " " using command down' in space.call_args[0][0]

    enter = _hotkey_on_darwin(keyboard, "enter", "command")
    assert 'keystroke "\n" using command down' in enter.call_args[0][0]


def test_down_and_up_delegate():
    """Keyboard.down()/up() call pyautogui.keyDown()/keyUp()."""
    keyboard = _make_keyboard()

    with mock.patch("interpreter.core.computer.keyboard.keyboard.pyautogui", mock.Mock()) as pyautogui:
        with mock.patch("interpreter.core.computer.keyboard.keyboard.time.sleep"):
            keyboard.down("a")
            keyboard.up("a")

    pyautogui.keyDown.assert_called_once_with("a")
    pyautogui.keyUp.assert_called_once_with("a")


def test_write_ignores_missing_clipboard():
    """Keyboard.write() tolerates a clipboard whose view() fails (it still copies
    and pastes the text, and skips restoring the prior history)."""
    computer = SimpleNamespace(clipboard=mock.Mock(view=mock.Mock(side_effect=AttributeError("no clipboard"))))
    keyboard = _make_keyboard(computer)

    with mock.patch("interpreter.core.computer.keyboard.keyboard.pyautogui", mock.Mock()):
        with mock.patch("interpreter.core.computer.keyboard.keyboard.time.sleep"):
            keyboard.write("hi")

    computer.clipboard.copy.assert_called_once_with("hi")
    computer.clipboard.paste.assert_called_once_with()
