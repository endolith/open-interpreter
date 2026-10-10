import datetime
from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.calendar.calendar import Calendar


def test_get_events_non_macos():
    """get_events() returns an unsupported-platform message on non-macOS systems."""
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.calendar.calendar.platform.system",
        return_value="Linux",
    ):
        assert cal.get_events() == "This method is only supported on MacOS"


def test_get_events_macos_runs_applescript():
    """On macOS, get_events() returns the output from run_applescript_capture."""
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.calendar.calendar.platform.system",
        return_value="Darwin",
    ):
        with mock.patch(
            "interpreter.core.computer.calendar.calendar.run_applescript_capture",
            return_value=("Meeting at 3pm", ""),
        ):
            result = cal.get_events()
    assert result == "Meeting at 3pm"


def test_create_event_non_macos():
    """create_event() reports MacOS-only support when called on non-macOS platforms."""
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.calendar.calendar.platform.system",
        return_value="Linux",
    ):
        result = cal.create_event(
            "Title",
            datetime.datetime(2024, 1, 1, 9, 0),
            datetime.datetime(2024, 1, 1, 10, 0),
        )
    assert result == "This method is only supported on MacOS"


def test_get_first_calendar_returns_stripped_name():
    """get_first_calendar() unpacks the capture tuple and strips it (issue #426).

    run_applescript_capture returns (stdout, stderr); the old code treated the
    single return value as a tuple and stdout[0] yielded one character, so the
    "default calendar" was a single letter and every event lookup missed.
    """
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.calendar.calendar.platform.system",
        return_value="Darwin",
    ):
        with mock.patch(
            "interpreter.core.computer.calendar.calendar.run_applescript_capture",
            return_value=("Work\n", ""),
        ):
            assert cal.get_first_calendar() == "Work"


def test_get_first_calendar_returns_none_when_empty():
    """No captured stdout means no default calendar, and None is returned."""
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.calendar.calendar.platform.system",
        return_value="Darwin",
    ):
        with mock.patch(
            "interpreter.core.computer.calendar.calendar.run_applescript_capture",
            return_value=("", ""),
        ):
            assert cal.get_first_calendar() is None
