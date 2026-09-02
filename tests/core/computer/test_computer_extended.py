"""Tests for the Computer orchestrator class.

The Computer class coordinates all subsystems (mouse, keyboard, display, etc.)
and generates the API system message. These tests cover the constructor,
tool discovery, shortcut methods, and serialization.
"""

import json
from unittest import mock

import pytest

from interpreter.core.computer.computer import Computer


def _make_computer(interpreter=None):
    """Build a Computer with a mocked interpreter to avoid subsystem side effects."""
    if interpreter is None:
        interpreter = mock.MagicMock()
        interpreter.max_output = 2800
    return Computer(interpreter)


def test_computer_initializes_all_subsystems():
    """Computer.__init__ creates all expected subsystem instances."""
    computer = _make_computer()
    assert computer.terminal is not None
    assert computer.mouse is not None
    assert computer.keyboard is not None
    assert computer.display is not None
    assert computer.clipboard is not None
    assert computer.mail is not None
    assert computer.sms is not None
    assert computer.calendar is not None
    assert computer.contacts is not None
    assert computer.browser is not None
    assert computer.os is not None
    assert computer.vision is not None
    assert computer.skills is not None
    assert computer.docs is not None
    assert computer.ai is not None
    assert computer.files is not None


def test_computer_mirrors_interpreter_max_output():
    """Computer.max_output mirrors the interpreter's max_output at construction."""
    interpreter = mock.MagicMock()
    interpreter.max_output = 5000
    computer = Computer(interpreter)
    assert computer.max_output == 5000


def test_computer_system_message_contains_computer_api():
    """The generated system message includes the computer API tools listing."""
    computer = _make_computer()
    assert "computer" in computer.system_message.lower()
    assert "THE COMPUTER API" in computer.system_message


def test_computer_import_computer_api_defaults_false():
    """import_computer_api defaults to False and tracks import state."""
    computer = _make_computer()
    assert computer.import_computer_api is False
    assert computer._has_imported_computer_api is False


def test_languages_property_shortcut():
    """computer.languages is a shortcut to computer.terminal.languages."""
    computer = _make_computer()
    terminal_languages = computer.terminal.languages
    assert computer.languages is terminal_languages


def test_languages_setter_shortcut():
    """Setting computer.languages updates computer.terminal.languages."""
    computer = _make_computer()
    new_languages = ["python", "shell"]
    computer.languages = new_languages
    assert computer.terminal.languages == new_languages


def test_run_shortcut_delegates_to_terminal():
    """computer.run() is a shortcut for computer.terminal.run()."""
    computer = _make_computer()
    with mock.patch.object(computer.terminal, "run", return_value="result") as mock_run:
        result = computer.run("python", "print('hi')")
    mock_run.assert_called_once_with("python", "print('hi')")
    assert result == "result"


def test_exec_shortcut_runs_shell():
    """computer.exec() runs shell code via terminal.run."""
    computer = _make_computer()
    with mock.patch.object(computer.terminal, "run", return_value="ok") as mock_run:
        computer.exec("ls -la")
    mock_run.assert_called_once_with("shell", "ls -la")


def test_stop_shortcut_delegates_to_terminal():
    """computer.stop() is a shortcut for computer.terminal.stop()."""
    computer = _make_computer()
    with mock.patch.object(computer.terminal, "stop") as mock_stop:
        computer.stop()
    mock_stop.assert_called_once()


def test_terminate_shortcut_delegates_to_terminal():
    """computer.terminate() is a shortcut for computer.terminal.terminate()."""
    computer = _make_computer()
    with mock.patch.object(computer.terminal, "terminate") as mock_terminate:
        computer.terminate()
    mock_terminate.assert_called_once()


def test_screenshot_shortcut_delegates_to_display():
    """computer.screenshot() is a shortcut for computer.display.screenshot()."""
    computer = _make_computer()
    with mock.patch.object(computer.display, "screenshot", return_value="img") as mock_shot:
        result = computer.screenshot()
    mock_shot.assert_called_once()
    assert result == "img"


def test_view_shortcut_delegates_to_display():
    """computer.view() is a shortcut for computer.display.screenshot()."""
    computer = _make_computer()
    with mock.patch.object(computer.display, "screenshot", return_value="img") as mock_shot:
        result = computer.view()
    mock_shot.assert_called_once()
    assert result == "img"


def test_get_all_computer_tools_list():
    """_get_all_computer_tools_list returns all subsystems in expected order."""
    computer = _make_computer()
    tools = computer._get_all_computer_tools_list()
    tool_names = [t.__class__.__name__.lower() for t in tools]
    expected = [
        "mouse", "keyboard", "display", "clipboard", "mail", "sms",
        "calendar", "contacts", "browser", "os", "vision", "skills",
        "docs", "ai", "files",
    ]
    assert tool_names == expected


def test_extract_tool_info_formats_signatures():
    """_extract_tool_info formats method signatures as computer.<name>(<params>)."""
    computer = _make_computer()

    class FakeTool:
        def do_something(self, arg1, arg2="default"):
            """Do something useful."""
            pass

    info = computer._extract_tool_info(FakeTool())
    assert info["signature"] == "FakeTool"
    assert len(info["methods"]) == 1
    assert "computer.faketool.do_something" in info["methods"][0]["signature"]
    assert "Do something useful" in info["methods"][0]["description"]


def test_extract_tool_info_browser_skips_driver_methods():
    """Browser tool info excludes methods containing 'driver' in their name."""
    computer = _make_computer()

    class Browser:
        def search(self, query):
            """Search the web."""
            pass

        def _driver_setup(self):
            """Should be skipped."""
            pass

    info = computer._extract_tool_info(Browser())
    method_names = [m["signature"] for m in info["methods"]]
    assert any("search" in s for s in method_names)
    assert not any("driver" in s for s in method_names)


def test_to_dict_filters_non_serializable():
    """to_dict returns only JSON-serializable attributes."""
    computer = _make_computer()
    computer.emit_images = True
    result = computer.to_dict()
    assert isinstance(result, dict)
    assert result["emit_images"] is True
    json.dumps(result)


def test_load_dict_sets_matching_attributes():
    """load_dict sets attributes that already exist on the Computer."""
    computer = _make_computer()
    original_api_base = computer.api_base
    computer.load_dict({"api_base": "http://localhost:8000"})
    assert computer.api_base == "http://localhost:8000"


def test_load_dict_ignores_unknown_keys():
    """load_dict ignores keys that don't exist as Computer attributes."""
    computer = _make_computer()
    computer.load_dict({"unknown_key": "value", "api_base": "http://localhost"})
    assert not hasattr(computer, "unknown_key")
    assert computer.api_base == "http://localhost"


def test_save_skills_defaults_true():
    """save_skills defaults to True."""
    computer = _make_computer()
    assert computer.save_skills is True


def test_import_skills_defaults_false():
    """import_skills defaults to False and tracks import state."""
    computer = _make_computer()
    assert computer.import_skills is False
    assert computer._has_imported_skills is False


def test_api_base_defaults_to_openinterpreter():
    """api_base defaults to the Open Interpreter hosted API."""
    computer = _make_computer()
    assert computer.api_base == "https://api.openinterpreter.com/v0"


def test_emit_images_defaults_true():
    """emit_images defaults to True."""
    computer = _make_computer()
    assert computer.emit_images is True
