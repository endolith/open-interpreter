import datetime
from types import SimpleNamespace
from unittest import mock

import subprocess

import pytest

from interpreter.core.computer.calendar.calendar import Calendar


DARWIN = "interpreter.core.computer.calendar.calendar.platform.system"
CAPTURE = "interpreter.core.computer.calendar.calendar.run_applescript_capture"
RUN = "interpreter.core.computer.calendar.calendar.run_applescript"


def _darwin():
    return mock.patch(DARWIN, return_value="Darwin")


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


def test_get_events_defaults_end_date_to_start_date():
    """Without end_date, the script queries midnight-to-midnight one day."""
    cal = Calendar(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("events", "")) as capture:
            assert cal.get_events(start_date=datetime.date(2024, 3, 5)) == ("events")
    script = capture.call_args[0][0]
    assert "makeDate(2024, 03, 05, 0, 0, 0)" in script
    assert "makeDate(2024, 03, 05, 23, 59, 59)" in script


def test_get_events_custom_range_uses_both_dates():
    """A custom range formats each bound into its own makeDate call."""
    cal = Calendar(computer=SimpleNamespace())
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
    cal = Calendar(computer=SimpleNamespace())
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
    cal = Calendar(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "boom")):
            assert cal.get_events() == "boom"


def test_create_event_success_names_the_calendar():
    """A created event reports which calendar it landed in."""
    cal = Calendar(computer=SimpleNamespace())
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

    Mocks get_first_calendar() returning None, which is the empty-capture
    result once get_first_calendar() unpacks the (stdout, stderr) tuple
    correctly (issue #426). Today an empty capture instead yields "" (the
    tuple's stdout element, stripped), so None is not yet the value a real
    empty lookup produces — the empty-string case below pins that.
    """
    cal = Calendar(computer=SimpleNamespace())
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


@pytest.mark.xfail(reason="create_event() checks `calendar is None`, so the empty string a real empty lookup returns today (issue #426) is used as a calendar name instead of aborting")
def test_create_event_without_calendar_aborts_when_default_is_empty():
    """An empty discovered calendar name aborts instead of creating in "".

    A real empty lookup returns "" today (get_first_calendar()'s tuple
    indexing strips an empty stdout to "" rather than returning None), and
    create_event() only rejects None, so this currently proceeds to create
    an event in a calendar named "". Until the production check rejects the
    empty result, this documents the reachable abort that should happen.
    """
    cal = Calendar(computer=SimpleNamespace())
    with _darwin():
        with mock.patch.object(cal, "get_first_calendar", return_value=""):
            with mock.patch(RUN) as run:
                result = cal.create_event(
                    "Standup",
                    datetime.datetime(2024, 1, 1, 9, 0),
                    datetime.datetime(2024, 1, 1, 9, 30),
                )
    assert result == ("Can't find a default calendar. Please try again and specify a calendar name.")
    run.assert_not_called()


def test_create_event_uses_discovered_default_calendar():
    """With no calendar given, the first calendar becomes the target."""
    cal = Calendar(computer=SimpleNamespace())
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
    cal = Calendar(computer=SimpleNamespace())
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
    cal = Calendar(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE) as capture:
            assert cal.delete_event(None, datetime.datetime(2024, 1, 1)) == ("Event title and start date are required")
            assert cal.delete_event("X", None) == ("Event title and start date are required")
    capture.assert_not_called()


def test_delete_event_non_macos():
    """delete_event() refuses off macOS instead of running AppleScript."""
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(DARWIN, return_value="Linux"):
        assert cal.delete_event("X", datetime.datetime(2024, 1, 1)) == ("This method is only supported on MacOS")


def test_delete_event_without_calendar_aborts_when_no_default():
    """With no calendar and no discoverable default, nothing is deleted."""
    cal = Calendar(computer=SimpleNamespace())
    with _darwin():
        with mock.patch.object(cal, "get_first_calendar", return_value=""):
            with mock.patch(CAPTURE) as capture:
                assert cal.delete_event("X", datetime.datetime(2024, 1, 1)) == (
                    "Can't find a default calendar. Please try again and specify a calendar name."
                )
    capture.assert_not_called()


def test_delete_event_success_path_returns_stdout_verbatim():
    """A clean run returns the script's own output text.

    `stderr, stdout = run_applescript_capture(script)` unpacks the
    (stdout, stderr) tuple backwards, so the "stdout" branch below actually
    sees stderr — but with empty stderr the first branch is falsy and the
    `elif stderr:` branch (holding real stdout) matches "successfully" and
    returns it whole. Tracked as an issue; pinned until fixed.
    """
    cal = Calendar(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("Event deleted successfully.", "")):
            assert (
                cal.delete_event("X", datetime.datetime(2024, 1, 1), calendar="Work") == "Event deleted successfully."
            )


@pytest.mark.xfail(reason="delete_event() unpacks the (stdout, stderr) tuple backwards (issue #410), so stderr lands in the stdout slot and only its first character is returned")
def test_delete_event_stderr_is_reported_as_the_error():
    """A real stderr is reported as the deletion error, not truncated.

    Once delete_event() unpacks run_applescript_capture() as (stdout,
    stderr), an empty stdout with stderr "execution error" falls into the
    error branch and is wrapped. Today the swapped unpacking puts the stderr
    text in the stdout slot and stdout[0] returns its first character,
    so this currently returns "e".
    """
    cal = Calendar(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "execution error")):
            assert cal.delete_event("X", datetime.datetime(2024, 1, 1), calendar="Work") == (
                "Error deleting event: execution error"
            )


@pytest.mark.xfail(reason="delete_event() unpacks the (stdout, stderr) tuple backwards and indexes stdout[0] (issue #410): ordinary stdout is treated as stderr and wrapped as an error, and fixing only the unpacking would still return just the first character")
def test_delete_event_returns_ordinary_stdout_verbatim():
    """Ordinary stdout is returned as-is, not wrapped as an error message.

    Once delete_event() unpacks run_applescript_capture() as (stdout,
    stderr) and returns the full stdout, a non-empty stdout is returned
    verbatim by the first branch. Today the swapped unpacking puts the
    stdout text in the stderr slot and it comes back as
    "Error deleting event: ..."; fixing only the unpacking would still
    return just stdout[0].strip(), the first character.
    """
    cal = Calendar(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("some output", "")):
            assert cal.delete_event("X", datetime.datetime(2024, 1, 1), calendar="Work") == "some output"


def test_delete_event_silent_run_returns_unknown_error():
    """Empty stdout and stderr fall through to the unknown-error message."""
    cal = Calendar(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "")):
            assert cal.delete_event("X", datetime.datetime(2024, 1, 1), calendar="Work") == (
                "Unknown error deleting event. Please check event title and date."
            )


def test_get_first_calendar_strips_the_first_tuple_item():
    """The (stdout, stderr) tuple is truthy, so [0] is stdout, stripped."""
    cal = Calendar(computer=SimpleNamespace())
    with mock.patch(CAPTURE, return_value=("  Work\n", "")):
        assert cal.get_first_calendar() == "Work"
