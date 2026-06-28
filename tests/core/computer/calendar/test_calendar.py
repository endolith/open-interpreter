import datetime
from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.calendar.calendar import Calendar


def test_get_events_non_macos():
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.calendar.calendar.platform.system",
        return_value="Linux",
    ):
        assert cal.get_events() == "This method is only supported on MacOS"


def test_get_events_macos_runs_applescript():
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
    assert "MacOS" in result
