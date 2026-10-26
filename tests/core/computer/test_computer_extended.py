"""Tests for the Toolbox orchestrator class.

The Toolbox class coordinates all subsystems (mouse, keyboard, display, etc.)
and generates the API system message. These tests cover the constructor,
tool discovery, shortcut methods, and serialization.
"""

import json
import platform
from unittest import mock

import pytest

from interpreter.core.toolbox.toolbox import Toolbox


def _make_toolbox(interpreter=None):
    """Build a Toolbox with a mocked interpreter to avoid subsystem side effects."""
    if interpreter is None:
        interpreter = mock.MagicMock()
        interpreter.max_output = 2800
    return Toolbox(interpreter)


def test_toolbox_initializes_all_subsystems():
    """Toolbox.__init__ creates all expected subsystem instances."""
    toolbox = _make_toolbox()
    assert toolbox.mouse is not None
    assert toolbox.keyboard is not None
    assert toolbox.display is not None
    assert toolbox.clipboard is not None
    assert toolbox.mail is not None
    assert toolbox.sms is not None
    assert toolbox.calendar is not None
    assert toolbox.contacts is not None
    assert toolbox.browser is not None
    assert toolbox.os is not None
    assert toolbox.web is not None
    assert toolbox.vision is not None
    assert toolbox.skills is not None
    assert toolbox.ai is not None
    assert toolbox.ai2 is not None
    assert toolbox.files is not None


def test_toolbox_mirrors_interpreter_max_output():
    """Toolbox.max_output mirrors the interpreter's max_output at construction."""
    interpreter = mock.MagicMock()
    interpreter.max_output = 5000
    toolbox = Toolbox(interpreter)
    assert toolbox.max_output == 5000


def test_toolbox_system_message_contains_toolbox_api():
    """The generated system message includes the toolbox API tools listing."""
    toolbox = _make_toolbox()
    assert "toolbox" in toolbox.system_message.lower()
    assert "The `toolbox` API" in toolbox.system_message


def test_toolbox_import_toolbox_api_defaults_false():
    """import_toolbox_api defaults to False and tracks import state."""
    toolbox = _make_toolbox()
    assert toolbox.import_toolbox_api is False
    assert toolbox._has_imported_toolbox_api is False
    # Backward-compatible alias used by older profiles.
    assert toolbox.import_computer_api is False


def test_languages_property_shortcut():
    """toolbox.languages is a shortcut to interpreter.terminal.languages."""
    toolbox = _make_toolbox()
    terminal_languages = toolbox.interpreter.terminal.languages
    assert toolbox.languages is terminal_languages


def test_languages_setter_shortcut():
    """Setting toolbox.languages updates interpreter.terminal.languages."""
    interpreter = mock.MagicMock()
    interpreter.max_output = 2800
    interpreter.terminal.languages = ["python"]
    toolbox = Toolbox(interpreter)
    toolbox.languages = ["python", "bash"]
    assert interpreter.terminal.languages == ["python", "bash"]


def test_run_shortcut_delegates_to_terminal():
    """toolbox.run() is a shortcut for interpreter.terminal.run()."""
    toolbox = _make_toolbox()
    with mock.patch.object(
        toolbox.interpreter.terminal, "run", return_value="result"
    ) as mock_run:
        result = toolbox.run("python", "print('hi')")
    mock_run.assert_called_once_with("python", "print('hi')")
    assert result == "result"


def test_exec_shortcut_runs_bash():
    """toolbox.exec() runs bash code via interpreter.terminal.run."""
    toolbox = _make_toolbox()
    with mock.patch.object(
        toolbox.interpreter.terminal, "run", return_value="ok"
    ) as mock_run:
        toolbox.exec("ls -la")
    mock_run.assert_called_once_with("bash", "ls -la")


def test_stop_shortcut_delegates_to_terminal():
    """toolbox.stop() is a shortcut for interpreter.terminal.stop()."""
    toolbox = _make_toolbox()
    with mock.patch.object(toolbox.interpreter.terminal, "stop") as mock_stop:
        toolbox.stop()
    mock_stop.assert_called_once()


def test_terminate_shortcut_delegates_to_terminal():
    """toolbox.terminate() is a shortcut for interpreter.terminal.terminate()."""
    toolbox = _make_toolbox()
    with mock.patch.object(toolbox.interpreter.terminal, "terminate") as mock_terminate:
        toolbox.terminate()
    mock_terminate.assert_called_once()


def test_screenshot_shortcut_delegates_to_display():
    """toolbox.screenshot() is a shortcut for toolbox.display.screenshot()."""
    toolbox = _make_toolbox()
    with mock.patch.object(
        toolbox.display, "screenshot", return_value="img"
    ) as mock_shot:
        result = toolbox.screenshot()
    mock_shot.assert_called_once()
    assert result == "img"


def test_view_shortcut_delegates_to_display():
    """toolbox.view() is a shortcut for toolbox.display.screenshot()."""
    toolbox = _make_toolbox()
    with mock.patch.object(
        toolbox.display, "screenshot", return_value="img"
    ) as mock_shot:
        result = toolbox.view()
    mock_shot.assert_called_once()
    assert result == "img"


def test_get_all_toolbox_tools_list():
    """_get_all_toolbox_tools_list returns all subsystems in expected order."""
    toolbox = _make_toolbox()
    tools = toolbox._get_all_toolbox_tools_list()
    tool_names = [t.__class__.__name__.lower() for t in tools]
    expected = [
        "mouse", "keyboard", "display", "clipboard", "browser", "os",
        "web", "vision", "skills", "ai", "ai2", "files",
    ]
    if platform.system() == "Darwin":
        expected = expected[:4] + ["mail", "sms", "calendar", "contacts"] + expected[4:]
    assert tool_names == expected


def test_extract_tool_info_formats_signatures():
    """_extract_tool_info formats method signatures as toolbox.<name>(<params>)."""
    toolbox = _make_toolbox()

    class FakeTool:
        def do_something(self, arg1, arg2="default"):
            """Do something useful."""
            pass

    info = toolbox._extract_tool_info(FakeTool())
    assert info["signature"] == "FakeTool"
    assert len(info["methods"]) == 1
    assert "toolbox.faketool.do_something" in info["methods"][0]["signature"]
    assert "Do something useful" in info["methods"][0]["description"]


def test_extract_tool_info_browser_skips_driver_methods():
    """Browser tool info excludes methods containing 'driver' in their name."""
    toolbox = _make_toolbox()

    class Browser:
        def search(self, query):
            """Search the web."""
            pass

        def _driver_setup(self):
            """Should be skipped."""
            pass

    info = toolbox._extract_tool_info(Browser())
    method_names = [m["signature"] for m in info["methods"]]
    assert any("search" in s for s in method_names)
    assert not any("driver" in s for s in method_names)


def test_to_dict_filters_non_serializable():
    """to_dict returns only JSON-serializable attributes."""
    toolbox = _make_toolbox()
    toolbox.emit_images = True
    result = toolbox.to_dict()
    assert isinstance(result, dict)
    assert result["emit_images"] is True
    json.dumps(result)


def test_load_dict_sets_matching_attributes():
    """load_dict sets attributes that already exist on the Toolbox."""
    toolbox = _make_toolbox()
    original_api_base = toolbox.api_base
    toolbox.load_dict({"api_base": "http://localhost:8000"})
    assert toolbox.api_base == "http://localhost:8000"


def test_load_dict_ignores_unknown_keys():
    """load_dict ignores keys that don't exist as Toolbox attributes."""
    toolbox = _make_toolbox()
    toolbox.load_dict({"unknown_key": "value", "api_base": "http://localhost"})
    assert not hasattr(toolbox, "unknown_key")
    assert toolbox.api_base == "http://localhost"


def test_save_skills_defaults_true():
    """save_skills defaults to True."""
    toolbox = _make_toolbox()
    assert toolbox.save_skills is True


def test_import_skills_defaults_false():
    """import_skills defaults to False and tracks import state."""
    toolbox = _make_toolbox()
    assert toolbox.import_skills is False
    assert toolbox._has_imported_skills is False


def test_api_base_defaults_to_openinterpreter():
    """api_base defaults to the Open Interpreter hosted API."""
    toolbox = _make_toolbox()
    assert toolbox.api_base == "https://api.openinterpreter.com/v0"


def test_emit_images_defaults_true():
    """emit_images defaults to True."""
    toolbox = _make_toolbox()
    assert toolbox.emit_images is True
