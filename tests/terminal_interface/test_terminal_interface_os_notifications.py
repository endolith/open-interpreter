"""Tests for OS-mode active-line action notifications in terminal_interface."""

import os
from unittest import mock

import pytest

from interpreter.terminal_interface.terminal_interface import terminal_interface


def _make_interpreter():
    interpreter = mock.MagicMock()
    interpreter.auto_run = True
    interpreter.offline = True
    interpreter.messages = []
    interpreter.plain_text_display = False
    interpreter.os = True
    interpreter.safe_mode = "off"
    interpreter.verbose = False
    interpreter.multi_line = False
    interpreter.max_output = 2000
    interpreter.debug = False
    interpreter.llm.supports_vision = False
    interpreter.llm.vision_renderer = None
    interpreter.speak_messages = False
    return interpreter


def _active_line_interpreter(code, active_line=0):
    interpreter = _make_interpreter()

    def chat(message, display=False, stream=True):
        yield {"type": "code", "role": "assistant", "format": "python", "start": True}
        yield {"type": "code", "role": "assistant", "content": code}
        yield {"type": "console", "role": "computer", "format": "active_line", "content": active_line}

    interpreter.chat = chat
    return interpreter


def test_mouse_click_notification():
    """OS mode notifies when computer.mouse.click() is the active line."""
    code = "computer.mouse.click()\n"
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "click"))
    interpreter.computer.os.notify.assert_called_with("Clicking...")


def test_mouse_click_with_text_notification():
    """OS mode notifies with 'text' when clicking by text."""
    code = 'computer.mouse.click("Submit")\n'
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "click"))
    interpreter.computer.os.notify.assert_called_with("Clicking text...")


def test_mouse_click_with_icon_notification():
    """OS mode notifies with 'icon' when clicking by icon."""
    code = 'computer.mouse.click(icon="button.png")\n'
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "click"))
    interpreter.computer.os.notify.assert_called_with("Clicking icon...")


def test_mouse_move_notification():
    """OS mode notifies 'Mousing over' when moving without click in code."""
    code = 'computer.mouse.move("target")\n'
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "move"))
    interpreter.computer.os.notify.assert_called_with("Mousing over text...")


def test_mouse_move_with_click_in_code_notification():
    """OS mode notifies 'Clicking' when moving with click elsewhere in code."""
    code = 'computer.mouse.click()\ncomputer.mouse.move("target")\n'
    interpreter = _active_line_interpreter(code, active_line=1)
    list(terminal_interface(interpreter, "move"))
    interpreter.computer.os.notify.assert_called_with("Clicking text...")


def test_mouse_move_with_icon_notification():
    """OS mode notifies with 'icon' when moving to icon."""
    code = 'computer.mouse.move(icon="button.png")\n'
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "move"))
    interpreter.computer.os.notify.assert_called_with("Mousing over icon...")


def test_keyboard_hotkey_notification():
    """OS mode notifies when computer.keyboard.hotkey() is the active line."""
    code = 'computer.keyboard.hotkey("ctrl", "c")\n'
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "hotkey"))
    interpreter.computer.os.notify.assert_called_with('Pressing "ctrl", "c".')


def test_keyboard_press_notification():
    """OS mode notifies when computer.keyboard.press() is the active line."""
    code = 'computer.keyboard.press("enter")\n'
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "press"))
    interpreter.computer.os.notify.assert_called_with('Pressing "enter".')


def test_get_selected_text_notification():
    """OS mode notifies when computer.os.get_selected_text() is active."""
    code = "computer.os.get_selected_text()\n"
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "select"))
    interpreter.computer.os.notify.assert_called_with("Getting selected text.")


def test_screenshot_action_notification():
    """OS mode notifies 'Viewing screen...' for screenshot actions."""
    code = "computer.screenshot()\n"
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "screenshot"))
    interpreter.computer.os.notify.assert_called_with("Viewing screen...")


def test_display_view_notification():
    """OS mode notifies 'Viewing screen...' for computer.display.view()."""
    code = "computer.display.view()\n"
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "view"))
    interpreter.computer.os.notify.assert_called_with("Viewing screen...")


def test_keyboard_write_notification():
    """OS mode notifies with the text being typed."""
    code = 'computer.keyboard.write("hello world")\n'
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "type"))
    interpreter.computer.os.notify.assert_called_with('Typing "hello world".')


def test_no_notification_for_non_computer_actions():
    """OS mode does not notify for non-computer active lines."""
    code = 'print("hello")\n'
    interpreter = _active_line_interpreter(code)
    list(terminal_interface(interpreter, "print"))
    interpreter.computer.os.notify.assert_not_called()


def test_darwin_speak_path():
    """Darwin platform with speak_messages terminates prior subprocess and speaks."""
    import interpreter.terminal_interface.terminal_interface as ti

    interpreter = _make_interpreter()
    interpreter.os = True
    interpreter.speak_messages = True
    interpreter.messages = [{"role": "assistant", "content": "hello world"}]

    def chat(message, display=False, stream=True):
        yield {"type": "message", "role": "assistant", "start": True}
        yield {"type": "message", "role": "assistant", "content": "hello world"}
        yield {"type": "message", "role": "assistant", "end": True}

    interpreter.chat = chat

    with mock.patch.object(ti.platform, "system", return_value="Darwin"):
        with mock.patch.object(ti.subprocess, "Popen") as popen:
            list(terminal_interface(interpreter, "speak"))

    popen.assert_called_once_with(
        ["osascript", "-e", 'say "hello world" using "Fred"']
    )


def test_voice_subprocess_stored_for_termination():
    """Voice subprocess is stored so it can be terminated on next message."""
    import interpreter.terminal_interface.terminal_interface as ti

    interpreter = _make_interpreter()
    interpreter.os = True
    interpreter.speak_messages = True
    interpreter.messages = [{"role": "assistant", "content": "first message"}]

    def chat(message, display=False, stream=True):
        yield {"type": "message", "role": "assistant", "start": True}
        yield {"type": "message", "role": "assistant", "content": "first message"}
        yield {"type": "message", "role": "assistant", "end": True}

    interpreter.chat = chat

    with mock.patch.object(ti.platform, "system", return_value="Darwin"):
        with mock.patch.object(ti.subprocess, "Popen") as popen:
            list(terminal_interface(interpreter, "speak"))

    assert popen.return_value is not None
