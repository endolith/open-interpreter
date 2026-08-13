"""Characterization tests for ``computer.mouse``.

These mock pyautogui and the display subsystem so they run headless in CI.
They pin the current mouse API so any rename or rework trips loudly on drift.
"""

from types import SimpleNamespace
from unittest import mock

import pytest


def _make_mouse(computer=None):
    from interpreter.core.computer.mouse.mouse import Mouse

    if computer is None:
        computer = SimpleNamespace(
            display=SimpleNamespace(width=100, height=100),
            emit_images=False,
            verbose=False,
        )
    return Mouse(computer)


def _patch_pyautogui(**attrs):
    """Patch the module-level ``pyautogui`` name (it is ``None`` via lazy_import
    on machines without the package) with a mock and return the mock."""
    pyautogui = mock.Mock()
    for name, value in attrs.items():
        getattr(pyautogui, name).return_value = value
    patcher = mock.patch(
        "interpreter.core.computer.mouse.mouse.pyautogui", pyautogui
    )
    return patcher, pyautogui


def _display_with_find(computer, find_result):
    display = SimpleNamespace(width=100, height=100, screenshot=mock.Mock(return_value="shot"))
    display.find = mock.Mock(return_value=find_result)
    computer.display = display
    return computer


def test_scroll_delegates_to_pyautogui():
    """Mouse.scroll() forwards the click count to pyautogui.scroll."""
    mouse = _make_mouse()
    patcher, pyautogui = _patch_pyautogui()
    with patcher:
        mouse.scroll(3)
    pyautogui.scroll.assert_called_once_with(3)


def test_position_returns_pyautogui_position():
    """Mouse.position() returns the current pyautogui position."""
    mouse = _make_mouse()
    patcher, pyautogui = _patch_pyautogui()
    pyautogui.position.return_value = (5, 6)
    with patcher:
        assert mouse.position() == (5, 6)
    pyautogui.position.assert_called_once_with()


def test_position_error_is_wrapped_in_runtime_error():
    """Mouse.position() wraps pyautogui failures in a RuntimeError."""
    mouse = _make_mouse()
    patcher, pyautogui = _patch_pyautogui()
    pyautogui.position.side_effect = Exception("no display")
    with patcher:
        with pytest.raises(RuntimeError, match="no display"):
            mouse.position()


def test_move_rejects_too_many_positional_args():
    """Mouse.move() raises ValueError when given more than one positional coordinate."""
    mouse = _make_mouse()
    with pytest.raises(ValueError, match="Too many positional arguments"):
        mouse.move(1, 2)


def test_move_requires_a_target():
    """Mouse.move() raises ValueError when no text, icon, or coordinates are given."""
    mouse = _make_mouse()
    with pytest.raises(ValueError, match="Either text, icon, or both x and y"):
        mouse.move()


def test_move_xy_delegates_to_smooth_move_to():
    """Mouse.move(x=, y=) smooth-moves to the given pixel coordinates."""
    mouse = _make_mouse()
    with mock.patch(
        "interpreter.core.computer.mouse.mouse.smooth_move_to"
    ) as smooth_move_to:
        mouse.move(x=10, y=20)
    smooth_move_to.assert_called_once_with(10, 20)


def test_move_text_locates_and_moves_to_scaled_coordinates():
    """Mouse.move("text") finds the text via display.find and moves to its
    normalized coordinates scaled to the screen size."""
    computer = SimpleNamespace(emit_images=False, verbose=False, display=None)
    computer = _display_with_find(
        computer,
        [{"coordinates": (0.5, 0.5), "similarity": 1, "text": "hello"}],
    )
    mouse = _make_mouse(computer)

    with mock.patch(
        "interpreter.core.computer.mouse.mouse.smooth_move_to"
    ) as smooth_move_to:
        mouse.move("hello")

    computer.display.find.assert_called_once_with('"hello"', screenshot="shot")
    smooth_move_to.assert_called_once_with(50, 50)


def test_move_text_falls_back_to_icon_when_not_found():
    """Mouse.move("text") retries as an icon when the text is not on screen."""
    computer = SimpleNamespace(emit_images=False, verbose=False, display=None)
    display = SimpleNamespace(
        width=100,
        height=100,
        screenshot=mock.Mock(return_value="shot"),
        find=mock.Mock(side_effect=[[], [(0.5, 0.5)]]),
    )
    computer.display = display
    mouse = _make_mouse(computer)

    with mock.patch(
        "interpreter.core.computer.mouse.mouse.smooth_move_to"
    ) as smooth_move_to:
        mouse.move("hello")

    assert display.find.call_count == 2
    assert display.find.call_args_list[0] == mock.call('"hello"', screenshot="shot")
    # The icon fallback passes the screenshot positionally.
    assert display.find.call_args_list[1] == mock.call("hello", "shot")
    smooth_move_to.assert_called_once_with(50, 50)


def test_move_text_multiple_matches_raises():
    """Mouse.move("text") raises when the text is found multiple times."""
    computer = SimpleNamespace(emit_images=False, verbose=False, display=None)
    computer = _display_with_find(
        computer,
        [
            {"coordinates": (0.1, 0.1), "similarity": 1, "text": "hello"},
            {"coordinates": (0.9, 0.9), "similarity": 1, "text": "hello"},
        ],
    )
    mouse = _make_mouse(computer)

    with pytest.raises(ValueError, match="found multiple times"):
        mouse.move("hello")


def test_move_icon_moves_to_scaled_coordinates():
    """Mouse.move(icon=) finds the icon via display.find and moves to its
    normalized coordinates scaled to the screen size."""
    computer = SimpleNamespace(emit_images=False, verbose=False, display=None)
    computer = _display_with_find(computer, [(0.5, 0.5)])
    mouse = _make_mouse(computer)

    with mock.patch(
        "interpreter.core.computer.mouse.mouse.smooth_move_to"
    ) as smooth_move_to:
        mouse.move(icon='"folder"')

    computer.display.find.assert_called_once_with("folder", "shot")
    smooth_move_to.assert_called_once_with(50, 50)


def test_move_icon_multiple_matches_raises():
    """Mouse.move(icon=) raises when the icon is found multiple times."""
    computer = SimpleNamespace(emit_images=False, verbose=False, display=None)
    computer = _display_with_find(computer, [(0.1, 0.1), (0.9, 0.9)])
    mouse = _make_mouse(computer)

    with pytest.raises(ValueError, match="found multiple times"):
        mouse.move(icon="folder")


def test_click_moves_then_clicks():
    """Mouse.click() moves first, then clicks with the default button/timing."""
    mouse = _make_mouse()
    patcher, pyautogui = _patch_pyautogui()
    with mock.patch("interpreter.core.computer.mouse.mouse.smooth_move_to") as smooth_move_to:
        with patcher:
            mouse.click(x=5, y=7)

    smooth_move_to.assert_called_once_with(5, 7)
    pyautogui.click.assert_called_once_with(button="left", clicks=1, interval=0.1)


def test_click_without_target_just_clicks():
    """Mouse.click() clicks in place when given no target."""
    mouse = _make_mouse()
    patcher, pyautogui = _patch_pyautogui()
    with patcher:
        mouse.click()
    pyautogui.click.assert_called_once_with(button="left", clicks=1, interval=0.1)


def test_double_click_moves_then_double_clicks():
    """Mouse.double_click() moves first, then calls pyautogui.doubleClick."""
    mouse = _make_mouse()
    patcher, pyautogui = _patch_pyautogui()
    with mock.patch("interpreter.core.computer.mouse.mouse.smooth_move_to"):
        with patcher:
            mouse.double_click(x=1, y=2)
    pyautogui.doubleClick.assert_called_once_with(button="left", interval=0.1)


def test_triple_click_delegates():
    """Mouse.triple_click() calls pyautogui.tripleClick."""
    mouse = _make_mouse()
    patcher, pyautogui = _patch_pyautogui()
    with patcher:
        mouse.triple_click()
    pyautogui.tripleClick.assert_called_once_with(button="left", interval=0.1)


def test_right_click_delegates():
    """Mouse.right_click() calls pyautogui.rightClick."""
    mouse = _make_mouse()
    patcher, pyautogui = _patch_pyautogui()
    with patcher:
        mouse.right_click()
    pyautogui.rightClick.assert_called_once_with()


def test_down_and_up_delegate():
    """Mouse.down()/up() call pyautogui.mouseDown()/mouseUp()."""
    mouse = _make_mouse()
    patcher, pyautogui = _patch_pyautogui()
    with patcher:
        mouse.down()
        mouse.up()
    pyautogui.mouseDown.assert_called_once_with()
    pyautogui.mouseUp.assert_called_once_with()


def test_smooth_move_to_eases_then_lands_exactly():
    """smooth_move_to() interpolates along an ease-in-out curve and always ends
    exactly at the target coordinates."""
    from interpreter.core.computer.mouse.mouse import smooth_move_to

    fake_pyautogui = mock.Mock()
    fake_pyautogui.position.return_value = (0, 0)
    with mock.patch("interpreter.core.computer.mouse.mouse.pyautogui", fake_pyautogui):
        with mock.patch(
            "interpreter.core.computer.mouse.mouse.time.time",
            side_effect=[0, 0.5, 1.5, 3.0],
        ):
            smooth_move_to(10, 20, duration=2)

    assert fake_pyautogui.moveTo.call_count == 3
    assert fake_pyautogui.moveTo.call_args_list[-1] == mock.call(10, 20)
