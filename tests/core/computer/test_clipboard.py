from unittest import mock

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
