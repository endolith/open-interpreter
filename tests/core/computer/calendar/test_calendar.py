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


def test_delete_event_returns_stdout_without_indexing():
    """delete_event() returns the stripped stdout string (issue #410).

    run_applescript_capture returns (stdout, stderr), so the old unpack read
    the string as a sequence and stdout[0] yielded a single character.
    """
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.calendar.calendar.platform.system",
        return_value="Darwin",
    ):
        with (
            mock.patch(
                "interpreter.core.computer.calendar.calendar.run_applescript_capture",
                return_value=("Event deleted successfully.\n", ""),
            ),
            mock.patch.object(cal, "get_first_calendar", return_value="Work"),
        ):
            result = cal.delete_event("Title", datetime.datetime(2024, 1, 1, 9))
    assert result == "Event deleted successfully."


def test_delete_event_passes_success_stderr_through():
    """AppleScript reports success on stderr; that message still reaches the caller."""
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.calendar.calendar.platform.system",
        return_value="Darwin",
    ):
        with (
            mock.patch(
                "interpreter.core.computer.calendar.calendar.run_applescript_capture",
                return_value=("", "successfully deleted"),
            ),
            mock.patch.object(cal, "get_first_calendar", return_value="Work"),
        ):
            result = cal.delete_event("Title", datetime.datetime(2024, 1, 1, 9))
    assert result == "successfully deleted"


def test_delete_event_reports_real_error():
    """An unsuccessful stderr becomes a prefixed error message."""
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.calendar.calendar.platform.system",
        return_value="Darwin",
    ):
        with (
            mock.patch(
                "interpreter.core.computer.calendar.calendar.run_applescript_capture",
                return_value=("", "No matching event"),
            ),
            mock.patch.object(cal, "get_first_calendar", return_value="Work"),
        ):
            result = cal.delete_event("Title", datetime.datetime(2024, 1, 1, 9))
    assert "Error deleting event" in result
    assert "No matching event" in result


def test_delete_event_non_macos():
    """delete_event() reports MacOS-only support off a Mac."""
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.calendar.calendar.platform.system",
        return_value="Linux",
    ):
        assert cal.delete_event("Title", datetime.datetime(2024, 1, 1, 9)) == "This method is only supported on MacOS"
