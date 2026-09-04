"""Tests for get_active_window.

get_active_window queries the OS window manager for the currently focused
window, with one branch per platform (Windows via pygetwindow, macOS via
AppKit/Quartz, Linux via EWMH/Xlib). All platform imports are deferred inside
the branches so the module imports everywhere; the tests stub platform.system
and the deferred modules. Documenting current behavior only — no source
changes.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.computer.utils.get_active_window import get_active_window


def test_windows_returns_region_and_title():
    """Windows returns the active window bounds and title via pygetwindow."""
    win = SimpleNamespace(left=10, top=20, width=800, height=600, title="Notepad")
    gw = SimpleNamespace(getActiveWindow=lambda: win)
    with (
        mock.patch("platform.system", return_value="Windows"),
        mock.patch.dict(sys.modules, {"pygetwindow": gw}),
    ):
        assert get_active_window() == {
            "region": (10, 20, 800, 600),
            "title": "Notepad",
        }


def test_windows_no_active_window_returns_none():
    """Windows returns None when pygetwindow reports no active window."""
    gw = SimpleNamespace(getActiveWindow=lambda: None)
    with (
        mock.patch("platform.system", return_value="Windows"),
        mock.patch.dict(sys.modules, {"pygetwindow": gw}),
    ):
        assert get_active_window() is None


def test_darwin_returns_region_and_title():
    """macOS returns the bounds and title of the active app's window."""
    bounds = {"x": 0, "y": 25, "width": 1440, "height": 875}
    appkit = SimpleNamespace(
        NSWorkspace=SimpleNamespace(
            sharedWorkspace=lambda: SimpleNamespace(activeApplication=lambda: {"NSApplicationName": "Safari"})
        )
    )
    quartz = SimpleNamespace(
        CGWindowListCopyWindowInfo=lambda _opt, _wid: [
            {
                "kCGWindowOwnerName": "Safari",
                "kCGWindowBounds": bounds,
                "kCGWindowName": "Example",
            }
        ],
        kCGNullWindowID=0,
        kCGWindowListOptionOnScreenOnly=0,
    )
    with (
        mock.patch("platform.system", return_value="Darwin"),
        mock.patch.dict(sys.modules, {"AppKit": appkit, "Quartz": quartz}),
    ):
        assert get_active_window() == {"region": bounds, "title": "Example"}


def test_darwin_no_matching_window_returns_none():
    """macOS returns None when no on-screen window belongs to the active app."""
    appkit = SimpleNamespace(
        NSWorkspace=SimpleNamespace(
            sharedWorkspace=lambda: SimpleNamespace(activeApplication=lambda: {"NSApplicationName": "Safari"})
        )
    )
    quartz = SimpleNamespace(
        CGWindowListCopyWindowInfo=lambda _opt, _wid: [{"kCGWindowOwnerName": "Finder", "kCGWindowBounds": {}}],
        kCGNullWindowID=0,
        kCGWindowListOptionOnScreenOnly=0,
    )
    with (
        mock.patch("platform.system", return_value="Darwin"),
        mock.patch.dict(sys.modules, {"AppKit": appkit, "Quartz": quartz}),
    ):
        assert get_active_window() is None


def test_darwin_missing_window_name_defaults_unknown():
    """macOS falls back to 'Unknown' when the window has no name."""
    bounds = {"x": 0, "y": 0, "width": 100, "height": 100}
    appkit = SimpleNamespace(
        NSWorkspace=SimpleNamespace(
            sharedWorkspace=lambda: SimpleNamespace(activeApplication=lambda: {"NSApplicationName": "Safari"})
        )
    )
    quartz = SimpleNamespace(
        CGWindowListCopyWindowInfo=lambda _opt, _wid: [{"kCGWindowOwnerName": "Safari", "kCGWindowBounds": bounds}],
        kCGNullWindowID=0,
        kCGWindowListOptionOnScreenOnly=0,
    )
    with (
        mock.patch("platform.system", return_value="Darwin"),
        mock.patch.dict(sys.modules, {"AppKit": appkit, "Quartz": quartz}),
    ):
        assert get_active_window() == {"region": bounds, "title": "Unknown"}


def test_linux_returns_region_and_title():
    """Linux returns the active window geometry and name via EWMH."""
    geom = SimpleNamespace(x=5, y=5, width=1024, height=768)
    win = SimpleNamespace(get_geometry=lambda: geom, get_wm_name=lambda: "Terminal")
    ewmh_mod = SimpleNamespace(EWMH=lambda: SimpleNamespace(getActiveWindow=lambda: win))
    xlib_display = SimpleNamespace(Display=object)
    with (
        mock.patch("platform.system", return_value="Linux"),
        mock.patch.dict(sys.modules, {"ewmh": ewmh_mod, "Xlib.display": xlib_display}),
    ):
        assert get_active_window() == {
            "region": (5, 5, 1024, 768),
            "title": "Terminal",
        }


def test_linux_no_active_window_returns_none():
    """Linux returns None when EWMH reports no active window."""
    ewmh_mod = SimpleNamespace(EWMH=lambda: SimpleNamespace(getActiveWindow=lambda: None))
    xlib_display = SimpleNamespace(Display=object)
    with (
        mock.patch("platform.system", return_value="Linux"),
        mock.patch.dict(sys.modules, {"ewmh": ewmh_mod, "Xlib.display": xlib_display}),
    ):
        assert get_active_window() is None


def test_unsupported_platform_exits(capsys):
    """An unrecognized platform prints a message and exits nonzero.

    get_active_window supports Windows, macOS, and Linux only; anything else
    is a hard error rather than a silent None so misconfiguration surfaces.
    """
    with mock.patch("platform.system", return_value="FreeBSD"):
        with pytest.raises(SystemExit) as exc_info:
            get_active_window()
    assert exc_info.value.code == 1
    assert "Unsupported platform" in capsys.readouterr().out
