import datetime
from types import SimpleNamespace
from unittest import mock

import subprocess

import pytest

from interpreter.core.toolbox.calendar.calendar import Calendar


DARWIN = "interpreter.core.toolbox.calendar.calendar.platform.system"
CAPTURE = "interpreter.core.toolbox.calendar.calendar.run_applescript_capture"
RUN = "interpreter.core.toolbox.calendar.calendar.run_applescript"


def _darwin():
    return mock.patch(DARWIN, return_value="Darwin")


def test_get_events_non_macos():
    """get_events() returns an unsupported-platform message on non-macOS systems."""
    cal = Calendar(toolbox=SimpleNamespace())
    with mock.patch(
        "interpreter.core.toolbox.calendar.calendar.platform.system",
        return_value="Linux",
    ):
        assert cal.get_events() == "This method is only supported on MacOS"


def test_get_events_macos_runs_applescript():
    """On macOS, get_events() returns the output from run_applescript_capture."""
    cal = Calendar(toolbox=SimpleNamespace())
    with mock.patch(
        "interpreter.core.toolbox.calendar.calendar.platform.system",
        return_value="Darwin",
    ):
        with mock.patch(
            "interpreter.core.toolbox.calendar.calendar.run_applescript_capture",
            return_value=("Meeting at 3pm", ""),
        ):
            result = cal.get_events()
    assert result == "Meeting at 3pm"


def test_create_event_non_macos():
    """create_event() reports MacOS-only support when called on non-macOS platforms."""
    cal = Calendar(toolbox=SimpleNamespace())
    with mock.patch(
        "interpreter.core.toolbox.calendar.calendar.platform.system",
        return_value="Linux",
    ):
        result = cal.create_event(
            "Title",
            datetime.datetime(2024, 1, 1, 9, 0),
            datetime.datetime(2024, 1, 1, 10, 0),
        )
    assert result == "This method is only supported on MacOS"


def test_get_events_defaults_end_date_to_start_date():
    """Without end_date, the script queries midnight-to-midnight one day."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("events", "")) as capture:
            assert cal.get_events(start_date=datetime.date(2024, 3, 5)) == ("events")
    script = capture.call_args[0][0]
    assert "makeDate(2024, 03, 05, 0, 0, 0)" in script
    assert "makeDate(2024, 03, 05, 23, 59, 59)" in script


def test_get_events_custom_range_uses_both_dates():
    """A custom range formats each bound into its own makeDate call."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("events", "")) as capture:
            cal.get_events(
                start_date=datetime.date(2024, 3, 5),
                end_date=datetime.date(2024, 3, 7),
            )
    script = capture.call_args[0][0]
    assert "makeDate(2024, 03, 05, 0, 0, 0)" in script
    assert "makeDate(2024, 03, 07, 23, 59, 59)" in script


def test_get_events_authorization_error_returns_guidance():
    """A Calendar automation denial becomes setup guidance, not raw stderr."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(
            CAPTURE,
            return_value=(
                "",
                "Not authorized to send Apple events to Calendar",
            ),
        ):
            result = cal.get_events()
    assert "Calendar access not authorized" in result
    assert "Automation" in result


def test_get_events_other_error_returns_stderr():
    """Non-auth errors pass through verbatim so the caller sees them."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "boom")):
            assert cal.get_events() == "boom"


def test_create_event_success_names_the_calendar():
    """A created event reports which calendar it landed in."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(RUN) as run:
            result = cal.create_event(
                "Standup",
                datetime.datetime(2024, 1, 1, 9, 0),
                datetime.datetime(2024, 1, 1, 9, 30),
                calendar="Work",
            )
    assert result == 'Event created successfully in the "Work" calendar.'
    script = run.call_args[0][0]
    assert 'summary:"Standup"' in script
    assert "makeDate(2024, 01, 01, 09, 00, 00)" in script
    assert "makeDate(2024, 01, 01, 09, 30, 00)" in script


def test_create_event_without_calendar_aborts_when_no_default():
    """With no calendar and no discoverable default, nothing is created.

    get_first_calendar() returns None on an empty capture (fixed in #427);
    this pins create_event's None check directly.
    """
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch.object(cal, "get_first_calendar", return_value=None):
            with mock.patch(RUN) as run:
                result = cal.create_event(
                    "Standup",
                    datetime.datetime(2024, 1, 1, 9, 0),
                    datetime.datetime(2024, 1, 1, 9, 30),
                )
    assert result == ("Can't find a default calendar. Please try again and specify a calendar name.")
    run.assert_not_called()


def test_create_event_without_calendar_aborts_when_default_is_empty():
    """An empty capture aborts instead of creating an event in ""."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "")):
            with mock.patch(RUN) as run:
                result = cal.create_event(
                    "Standup",
                    datetime.datetime(2024, 1, 1, 9, 0),
                    datetime.datetime(2024, 1, 1, 9, 30),
                )
    assert result == ("Can't find a default calendar. Please try again and specify a calendar name.")
    run.assert_not_called()


def test_get_first_calendar_returns_stripped_name():
    """get_first_calendar() unpacks the capture tuple and strips it (issue #426).

    run_applescript_capture returns (stdout, stderr); the old code bound the
    whole tuple to stdout, so `if stdout` tested a non-empty 2-tuple and was
    always true. The None branch was unreachable, and empty output returned
    "" instead of None (fixed in #427).
    """
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("Work\n", "")):
            assert cal.get_first_calendar() == "Work"


def test_get_first_calendar_returns_none_when_empty():
    """No captured stdout means no default calendar, and None is returned."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "")):
            assert cal.get_first_calendar() is None


def test_get_first_calendar_treats_whitespace_as_no_calendar():
    """Whitespace-only stdout means no calendar, not an empty calendar name.

    Stripping before the emptiness check keeps a whitespace-only result from
    becoming "", which create_event would treat as a calendar name.
    """
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("  \n", "")):
            assert cal.get_first_calendar() is None


def test_create_event_uses_discovered_default_calendar():
    """With no calendar given, the first calendar becomes the target."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch.object(cal, "get_first_calendar", return_value="Home"):
            with mock.patch(RUN) as run:
                result = cal.create_event(
                    "Standup",
                    datetime.datetime(2024, 1, 1, 9, 0),
                    datetime.datetime(2024, 1, 1, 9, 30),
                )
    assert result == 'Event created successfully in the "Home" calendar.'
    assert 'tell calendar "Home"' in run.call_args[0][0]


def test_create_event_failure_returns_the_error_text():
    """An osascript failure surfaces as the exception text, not a raise."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(
            RUN,
            side_effect=subprocess.CalledProcessError(1, "osascript", "bad date"),
        ):
            result = cal.create_event(
                "Standup",
                datetime.datetime(2024, 1, 1, 9, 0),
                datetime.datetime(2024, 1, 1, 9, 30),
                calendar="Work",
            )
    assert "non-zero exit status 1" in result


def test_delete_event_requires_title_and_date():
    """Missing title or date is rejected before any AppleScript runs."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE) as capture:
            assert cal.delete_event(None, datetime.datetime(2024, 1, 1)) == ("Event title and start date are required")
            assert cal.delete_event("X", None) == ("Event title and start date are required")
    capture.assert_not_called()


def test_delete_event_non_macos():
    """delete_event() refuses off macOS instead of running AppleScript."""
    cal = Calendar(toolbox=SimpleNamespace())
    with mock.patch(DARWIN, return_value="Linux"):
        assert cal.delete_event("X", datetime.datetime(2024, 1, 1)) == ("This method is only supported on MacOS")


def test_delete_event_without_calendar_aborts_when_no_default():
    """With no calendar and no discoverable default, nothing is deleted."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch.object(cal, "get_first_calendar", return_value=""):
            with mock.patch(CAPTURE) as capture:
                assert cal.delete_event("X", datetime.datetime(2024, 1, 1)) == (
                    "Can't find a default calendar. Please try again and specify a calendar name."
                )
    capture.assert_not_called()


def test_delete_event_success_path_returns_stdout_verbatim():
    """A clean run returns the script's own output text.

    delete_event() unpacks run_applescript_capture() as (stdout, stderr)
    (fixed in #425), so a non-empty stdout is returned verbatim by the
    first branch.
    """
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("Event deleted successfully.", "")):
            assert (
                cal.delete_event("X", datetime.datetime(2024, 1, 1), calendar="Work") == "Event deleted successfully."
            )


def test_delete_event_success_word_in_stderr_returns_stderr_verbatim():
    """A "successfully" stderr with empty stdout is returned as-is.

    Some Calendar backends report success on stderr; the `elif stderr:`
    branch returns that text whole instead of wrapping it as an error.
    Covers the inner branch the corrected unpacking would otherwise leave
    unexecuted.
    """
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "Event deleted successfully.")):
            assert (
                cal.delete_event("X", datetime.datetime(2024, 1, 1), calendar="Work") == "Event deleted successfully."
            )


def test_delete_event_stderr_is_reported_as_the_error():
    """A real stderr is reported as the deletion error, not truncated.

    delete_event() unpacks run_applescript_capture() as (stdout, stderr)
    (fixed in #425), so an empty stdout with stderr "execution error" falls
    into the error branch and is wrapped.
    """
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "execution error")):
            assert cal.delete_event("X", datetime.datetime(2024, 1, 1), calendar="Work") == (
                "Error deleting event: execution error"
            )


def test_delete_event_returns_ordinary_stdout_verbatim():
    """Ordinary stdout is returned as-is, not wrapped as an error message.

    delete_event() unpacks run_applescript_capture() as (stdout, stderr)
    and returns the full stdout (fixed in #425), so a non-empty stdout is
    returned verbatim by the first branch.
    """
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("some output", "")):
            assert cal.delete_event("X", datetime.datetime(2024, 1, 1), calendar="Work") == "some output"


def test_delete_event_silent_run_returns_unknown_error():
    """Empty stdout and stderr fall through to the unknown-error message."""
    cal = Calendar(toolbox=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "")):
            assert cal.delete_event("X", datetime.datetime(2024, 1, 1), calendar="Work") == (
                "Unknown error deleting event. Please check event title and date."
            )


