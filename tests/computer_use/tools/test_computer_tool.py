import asyncio
import base64
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

_tools = Path(__file__).resolve().parents[3] / "interpreter/computer_use/tools"

for mod_name, filename in [
    ("interpreter.computer_use.tools.base", "base.py"),
    ("interpreter.computer_use.tools.run", "run.py"),
]:
    spec = importlib.util.spec_from_file_location(mod_name, _tools / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

# computer.py imports pyautogui at module load; stub it so the module imports
# headless (pyautogui.size() needs a DISPLAY) without real input automation.
sys.modules["pyautogui"] = mock.MagicMock()

spec = importlib.util.spec_from_file_location("interpreter.computer_use.tools.computer", _tools / "computer.py")
_computer = importlib.util.module_from_spec(spec)
sys.modules["interpreter.computer_use.tools.computer"] = _computer
spec.loader.exec_module(_computer)

ComputerTool = _computer.ComputerTool
ScalingSource = _computer.ScalingSource
ToolError = sys.modules["interpreter.computer_use.tools.base"].ToolError
ToolResult = sys.modules["interpreter.computer_use.tools.base"].ToolResult
chunks = _computer.chunks
smooth_move_to = _computer.smooth_move_to


@pytest.fixture(autouse=True)
def _reset_pyautogui():
    """Give every test a deterministic, mock-controlled pyautogui with a known screen size."""
    _computer.pyautogui.reset_mock()
    _computer.pyautogui.size.return_value = (1920, 1080)
    yield


def _make_tool(scaling=True, width=1920, height=1080):
    """Build a ComputerTool without __init__, so tests never call pyautogui.size()."""
    tool = ComputerTool.__new__(ComputerTool)
    tool._scaling_enabled = scaling
    tool.width = width
    tool.height = height
    tool.display_num = None
    return tool


def test_chunks_splits_string_by_size():
    """chunks splits a string into fixed-width pieces, the last one shorter if the length is not a multiple."""
    assert chunks("abcdef", 2) == ["ab", "cd", "ef"]
    assert chunks("abcde", 2) == ["ab", "cd", "e"]


def test_chunks_empty_string_returns_empty():
    """chunks of an empty string produces no pieces rather than one empty string."""
    assert chunks("", 3) == []


def test_chunks_chunk_size_larger_than_string():
    """chunks returns the whole string as a single piece when the chunk size exceeds the string length."""
    assert chunks("hi", 5) == ["hi"]


def test_tool_metadata():
    """ComputerTool advertises the Anthropic computer-use tool name and api_type."""
    assert ComputerTool.name == "computer"
    assert ComputerTool.api_type == "computer_20241022"


def test_init_captures_screen_size():
    """ComputerTool.__init__ reads the screen size from pyautogui and stores width/height and display_num."""
    tool = ComputerTool()
    assert tool.width == 1920
    assert tool.height == 1080
    assert tool.display_num is None
    _computer.pyautogui.size.assert_called_once()


def test_scale_coordinates_disabled_passthrough():
    """With scaling disabled, scale_coordinates returns coordinates unchanged for both API and computer sources."""
    tool = _make_tool(scaling=False)
    assert tool.scale_coordinates(ScalingSource.COMPUTER, 500, 400) == (500, 400)
    assert tool.scale_coordinates(ScalingSource.API, 500, 400) == (500, 400)


def test_scale_coordinates_computer_scales_down():
    """Computer-source coordinates are scaled down to the nearest supported target resolution to shrink images sent to the model."""
    tool = _make_tool(scaling=True, width=1920, height=1080)
    assert tool.scale_coordinates(ScalingSource.COMPUTER, 1920, 1080) == (1366, 768)


def test_scale_coordinates_api_scales_up():
    """API-source coordinates are scaled up from the model's smaller coordinate space back to the physical screen."""
    tool = _make_tool(scaling=True, width=1920, height=1080)
    assert tool.scale_coordinates(ScalingSource.API, 1366, 768) == (1920, 1080)


def test_scale_coordinates_api_rejects_out_of_bounds():
    """API-source coordinates outside the physical screen raise ToolError because the model is pointing at a location it cannot see."""
    tool = _make_tool(scaling=True, width=1920, height=1080)
    with pytest.raises(ToolError, match="out of bounds"):
        tool.scale_coordinates(ScalingSource.API, 2000, 500)


def test_scale_coordinates_unknown_aspect_ratio_passthrough():
    """A screen whose aspect ratio matches no supported target is not scaled, preserving exact coordinates."""
    tool = _make_tool(scaling=True, width=3000, height=1000)
    assert tool.scale_coordinates(ScalingSource.COMPUTER, 1500, 500) == (1500, 500)
    assert tool.scale_coordinates(ScalingSource.API, 1500, 500) == (1500, 500)


def test_scale_coordinates_small_screen_passthrough():
    """A screen smaller than every scaling target (only a larger match exists) is left unscaled."""
    tool = _make_tool(scaling=True, width=800, height=600)
    assert tool.scale_coordinates(ScalingSource.COMPUTER, 400, 300) == (400, 300)


def test_options_reports_scaled_display_size():
    """options reports the scaled width/height plus display_number to the model, matching the coordinate space it will use."""
    tool = _make_tool(scaling=True, width=1920, height=1080)
    assert tool.options == {
        "display_width_px": 1366,
        "display_height_px": 768,
        "display_number": None,
    }


def test_to_params_includes_name_and_type():
    """to_params returns the Anthropic computer-use tool descriptor combining name, api_type, and options."""
    tool = _make_tool(scaling=True, width=1920, height=1080)
    params = tool.to_params()
    assert params["name"] == "computer"
    assert params["type"] == "computer_20241022"
    assert params["display_width_px"] == 1366


def test_call_mouse_move_requires_coordinate():
    """mouse_move without a coordinate raises ToolError because there is nowhere to move the pointer."""
    tool = _make_tool()
    with pytest.raises(ToolError, match="coordinate is required for mouse_move"):
        asyncio.run(tool(action="mouse_move"))


def test_call_left_click_drag_requires_coordinate():
    """left_click_drag without a coordinate raises ToolError because a drag needs a destination."""
    tool = _make_tool()
    with pytest.raises(ToolError, match="coordinate is required for left_click_drag"):
        asyncio.run(tool(action="left_click_drag"))


def test_call_key_requires_text():
    """key action without text raises ToolError because there is no key to press."""
    tool = _make_tool()
    with pytest.raises(ToolError, match="text is required for key"):
        asyncio.run(tool(action="key"))


def test_call_type_requires_text():
    """type action without text raises ToolError because there is nothing to type."""
    tool = _make_tool()
    with pytest.raises(ToolError, match="text is required for type"):
        asyncio.run(tool(action="type"))


def test_call_invalid_action_raises():
    """An unrecognized action raises ToolError instead of silently doing nothing."""
    tool = _make_tool()
    with pytest.raises(ToolError, match="Invalid action"):
        asyncio.run(tool(action="nonsense"))


def test_call_key_single_press_normalizes():
    """A single key is normalized (e.g. enter -> return) and pressed via pyautogui, followed by a screenshot."""
    tool = _make_tool()
    shot = mock.AsyncMock(return_value=ToolResult(output="shot"))
    with mock.patch.object(tool, "screenshot", new=shot):
        result = asyncio.run(tool(action="key", text="enter"))
    _computer.pyautogui.press.assert_called_once_with("return")
    assert result.output == "shot"


def test_call_key_removes_underscores():
    """Key names with underscores are normalized to pyautogui spellings (page_up -> pgup) before pressing."""
    tool = _make_tool()
    shot = mock.AsyncMock(return_value=ToolResult(output="shot"))
    with mock.patch.object(tool, "screenshot", new=shot):
        asyncio.run(tool(action="key", text="page_up"))
    _computer.pyautogui.press.assert_called_once_with("pgup")


def test_call_key_hotkey_uses_pyautogui():
    """Modifier-combined keys are dispatched through pyautogui.hotkey on non-macOS platforms."""
    tool = _make_tool()
    shot = mock.AsyncMock(return_value=ToolResult(output="shot"))
    with mock.patch.object(tool, "screenshot", new=shot):
        asyncio.run(tool(action="key", text="ctrl+c"))
    _computer.pyautogui.hotkey.assert_called_once_with("ctrl", "c")


def test_call_key_darwin_uses_applescript():
    """On macOS, super+ is rewritten to command+ and the hotkey is sent through osascript via os.system."""
    tool = _make_tool()
    shot = mock.AsyncMock(return_value=ToolResult(output="shot"))
    with (
        mock.patch("platform.system", return_value="Darwin"),
        mock.patch("os.system") as os_system,
        mock.patch.object(tool, "screenshot", new=shot),
    ):
        asyncio.run(tool(action="key", text="super+space"))
    os_system.assert_called_once()
    command = os_system.call_args.args[0]
    assert "osascript" in command
    assert "keystroke" in command
    assert "command" in command
    assert 'keystroke " " using command down' in command


def test_call_type_writes_text():
    """type writes the given text through pyautogui at the configured typing interval, then screenshots."""
    tool = _make_tool()
    shot = mock.AsyncMock(return_value=ToolResult(output="shot"))
    with mock.patch.object(tool, "screenshot", new=shot):
        result = asyncio.run(tool(action="type", text="hello"))
    _computer.pyautogui.write.assert_called_once_with("hello", interval=0.012)
    assert result.output == "shot"


def test_call_left_click():
    """left_click clicks with the left button through pyautogui and returns the follow-up screenshot."""
    tool = _make_tool()
    shot = mock.AsyncMock(return_value=ToolResult(output="shot"))
    with mock.patch.object(tool, "screenshot", new=shot):
        result = asyncio.run(tool(action="left_click"))
    _computer.pyautogui.click.assert_called_once_with(button="left")
    assert result.output == "shot"


def test_call_double_click_clicks_twice():
    """double_click issues two separate pyautogui clicks to emulate a double-click."""
    tool = _make_tool()
    shot = mock.AsyncMock(return_value=ToolResult(output="shot"))
    with mock.patch.object(tool, "screenshot", new=shot):
        asyncio.run(tool(action="double_click"))
    assert _computer.pyautogui.click.call_count == 2


def test_call_mouse_move_scales_and_smooth_moves():
    """mouse_move smooth-moves to the API coordinate scaled up to the physical screen, then screenshots."""
    tool = _make_tool()
    shot = mock.AsyncMock(return_value=ToolResult(output="shot"))
    with mock.patch.object(_computer, "smooth_move_to") as smooth, mock.patch.object(tool, "screenshot", new=shot):
        result = asyncio.run(tool(action="mouse_move", coordinate=(1366, 768)))
    smooth.assert_called_once_with(1920, 1080)
    assert result.output == "shot"


def test_call_left_click_drag_scales_and_drags():
    """left_click_drag smooth-moves to the scaled target and then drags there with the left button held."""
    tool = _make_tool()
    shot = mock.AsyncMock(return_value=ToolResult(output="shot"))
    with mock.patch.object(_computer, "smooth_move_to") as smooth, mock.patch.object(tool, "screenshot", new=shot):
        asyncio.run(tool(action="left_click_drag", coordinate=(1366, 768)))
    smooth.assert_called_once_with(1920, 1080)
    _computer.pyautogui.dragTo.assert_called_once_with(1920, 1080, button="left")


def test_call_cursor_position_skips_screenshot():
    """cursor_position returns the current pointer coordinates and does not take a screenshot."""
    tool = _make_tool(scaling=False)
    _computer.pyautogui.position.return_value = (10, 20)
    shot = mock.AsyncMock()
    with mock.patch.object(tool, "screenshot", new=shot):
        result = asyncio.run(tool(action="cursor_position"))
    assert result.output == "X=10,Y=20"
    shot.assert_not_called()


def test_screenshot_returns_base64_png(tmp_path):
    """screenshot saves the pyautogui capture to a temp PNG, base64-encodes it, and removes the temp file."""
    from PIL import Image

    _computer.pyautogui.screenshot.return_value = Image.new("RGB", (50, 50), "red")
    tool = _make_tool(scaling=False)
    with (
        mock.patch.object(_computer.tempfile, "gettempdir", return_value=str(tmp_path)),
        mock.patch.object(_computer, "uuid4", return_value=mock.MagicMock(hex="deadbeef")),
    ):
        result = asyncio.run(tool.screenshot())
    assert not (tmp_path / "screenshot_deadbeef.png").exists()
    assert result.base64_image
    data = base64.b64decode(result.base64_image)
    assert data.startswith(b"\x89PNG")
    _computer.pyautogui.screenshot.assert_called_once()


def test_shell_returns_command_output_and_screenshot():
    """shell runs the command and, by default, waits then attaches a screenshot to the result."""

    async def fake_run(command):
        return (0, "stdout text", "stderr text")

    tool = _make_tool()
    tool._screenshot_delay = 0
    shot = mock.AsyncMock(return_value=ToolResult(base64_image="imgdata"))
    with (
        mock.patch.object(_computer, "run", side_effect=fake_run) as mock_run,
        mock.patch.object(tool, "screenshot", new=shot),
    ):
        result = asyncio.run(tool.shell("echo hi"))
    mock_run.assert_awaited_once_with("echo hi")
    assert result.output == "stdout text"
    assert result.error == "stderr text"
    assert result.base64_image == "imgdata"
    shot.assert_awaited_once()


def test_shell_without_screenshot():
    """shell with take_screenshot=False runs the command and returns output without capturing a screenshot."""

    async def fake_run(command):
        return (0, "out", None)

    tool = _make_tool()
    shot = mock.AsyncMock()
    with (
        mock.patch.object(_computer, "run", side_effect=fake_run) as mock_run,
        mock.patch.object(tool, "screenshot", new=shot),
    ):
        result = asyncio.run(tool.shell("echo hi", take_screenshot=False))
    mock_run.assert_awaited_once_with("echo hi")
    assert result.output == "out"
    assert result.base64_image is None
    shot.assert_not_awaited()


def test_smooth_move_to_ends_at_target():
    """smooth_move_to eases the pointer along an easeInOutSine path and ends exactly at the requested target."""
    _computer.pyautogui.position.return_value = (0, 0)
    times = iter([0.0, 0.5, 1.0, 2.0])
    with mock.patch("time.time", side_effect=lambda: next(times)):
        smooth_move_to(100, 50, duration=1.0)
    calls = _computer.pyautogui.moveTo.call_args_list
    # at t=0.5 the eased fraction is (1-cos(pi/2))/2 = 0.5, so the pointer is halfway along the path
    assert any(abs(args[0] - 50) < 1e-9 and abs(args[1] - 25) < 1e-9 for args in [c.args for c in calls])
    assert calls[-1] == mock.call(100, 50)
    assert len(calls) > 1
