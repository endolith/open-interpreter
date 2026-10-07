from unittest import mock

import pytest

import interpreter.core.computer.clipboard.clipboard as clipboard_mod
from interpreter.core.computer.clipboard.clipboard import Clipboard


def test_clipboard_view_returns_pasted_content(monkeypatch):
    """Clipboard.view returns whatever pyperclip.paste provides."""
    computer = mock.MagicMock()
    fake_pyperclip = mock.MagicMock()
    fake_pyperclip.paste.return_value = "clipboard contents"
    monkeypatch.setattr(clipboard_mod, "pyperclip", fake_pyperclip)

    assert Clipboard(computer).view() == "clipboard contents"


def test_clipboard_copy_calls_pyperclip(monkeypatch):
    """Clipboard.copy(text) forwards the text to pyperclip.copy."""
    computer = mock.MagicMock()
    fake_pyperclip = mock.MagicMock()
    monkeypatch.setattr(clipboard_mod, "pyperclip", fake_pyperclip)

    Clipboard(computer).copy("some text")

    fake_pyperclip.copy.assert_called_once_with("some text")


def test_clipboard_copy_none_triggers_keyboard_hotkey(monkeypatch):
    """Clipboard.copy(None) performs a copy hotkey instead of using pyperclip."""
    computer = mock.MagicMock()
    fake_pyperclip = mock.MagicMock()
    monkeypatch.setattr(clipboard_mod, "pyperclip", fake_pyperclip)

    clip = Clipboard(computer)
    clip.copy(None)

    computer.keyboard.hotkey.assert_called_once_with(clip.modifier_key, "c")
    fake_pyperclip.copy.assert_not_called()


@pytest.mark.parametrize(
    "system,expected",
    [
        ("Darwin", "command"),
        ("Windows", "ctrl"),
        ("Linux", "ctrl"),
    ],
)
def test_modifier_key_matches_the_platform(monkeypatch, system, expected):
    """macOS pastes with command; Windows and Linux with control.

    The platform is read once in `__init__`, so a wrong chord here makes paste
    type the literal letter "v" instead of pasting — which reads as a broken
    clipboard rather than a wrong modifier. Faking `platform.system()` pins all
    three branches on any runner, instead of leaving macOS to whichever machine
    happens to execute it.
    """
    monkeypatch.setattr(clipboard_mod.platform, "system", lambda: system)

    assert Clipboard(mock.MagicMock()).modifier_key == expected


def test_paste_presses_the_modifier_with_v(monkeypatch):
    """paste() sends the paste chord for the current platform.

    It goes through the computer's keyboard rather than pyperclip, so it takes
    part in the same key handling as everything else the assistant types.
    """
    monkeypatch.setattr(clipboard_mod.platform, "system", lambda: "Linux")
    computer = mock.MagicMock()

    Clipboard(computer).paste()

    computer.keyboard.hotkey.assert_called_once_with("ctrl", "v")


def test_clipboard_copy_with_text_leaves_the_keyboard_alone(monkeypatch):
    """copy(text) uses pyperclip only, and must not also fire the copy chord.

    The existing copy tests cover each destination in isolation; this pins that
    knowing the text does *not* also simulate a keystroke, which would otherwise
    read the user's selection and overwrite what was asked for.
    """
    monkeypatch.setattr(clipboard_mod.platform, "system", lambda: "Linux")
    computer = mock.MagicMock()
    fake_pyperclip = mock.MagicMock()
    monkeypatch.setattr(clipboard_mod, "pyperclip", fake_pyperclip)

    Clipboard(computer).copy("some text")

    fake_pyperclip.copy.assert_called_once_with("some text")
    computer.keyboard.hotkey.assert_not_called()


def test_clipboard_paste_triggers_keyboard_hotkey():
    """Clipboard.paste() performs a paste hotkey with the platform modifier."""
    computer = mock.MagicMock()
    clip = Clipboard(computer)
    clip.paste()
    computer.keyboard.hotkey.assert_called_once_with(clip.modifier_key, "v")
