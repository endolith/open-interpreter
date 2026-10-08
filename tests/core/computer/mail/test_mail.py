from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.mail.mail import Mail


def test_mail_get_non_macos_returns_message():
    """Mail.get() returns an unsupported-platform message on non-macOS systems."""
    mail = Mail(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.mail.mail.platform.system", return_value="Linux"
    ):
        assert mail.get() == "This method is only supported on MacOS"


def test_mail_get_on_macos_runs_applescript():
    """On macOS, Mail.get() runs AppleScript limited to the requested message count."""
    mail = Mail(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.mail.mail.platform.system", return_value="Darwin"
    ):
        with mock.patch(
            "interpreter.core.computer.mail.mail.run_applescript_capture",
            return_value=("inbox data", ""),
        ) as capture:
            result = mail.get(number=2)
    capture.assert_called_once()
    script = capture.call_args[0][0]
    assert "repeat with i from 1 to 2" in script
    assert result == "inbox data"


def test_mail_get_stops_after_max_retries_on_shortage():
    """A persistent "can't get item" error stops at the retry limit (issue #409).

    Without the break inside the Can't-get-item branch, a non-numeric or
    non-shrinking error made the loop run until the retry budget ran out.
    """
    mail = Mail(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.mail.mail.platform.system", return_value="Darwin"
    ):
        with mock.patch(
            "interpreter.core.computer.mail.mail.run_applescript_capture",
            return_value=("", "Can’t get item 1 of"),
        ) as capture:
            result = mail.get(number=5)
    assert result is None
    # One call per iteration until the retry budget is exhausted, never more.
    assert 0 < capture.call_count <= 3


def test_mail_get_gives_up_on_unrecognized_error_without_output():
    """An unrecognized error and no output exits the loop instead of spinning (issue #409).

    Neither the Can't-get-item branch nor the stdout branch applies, so the
    loop body ends without a continue: the added break makes that explicit.
    """
    mail = Mail(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.mail.mail.platform.system", return_value="Darwin"
    ):
        with mock.patch(
            "interpreter.core.computer.mail.mail.run_applescript_capture",
            return_value=("", "Some unrelated applescript error"),
        ) as capture:
            result = mail.get()
    assert result is None
    capture.assert_called_once()


def test_mail_get_retries_with_available_count_then_returns():
    """When the error names a smaller available count, the retry returns that many."""
    mail = Mail(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.mail.mail.platform.system", return_value="Darwin"
    ):
        with mock.patch(
            "interpreter.core.computer.mail.mail.run_applescript_capture",
            side_effect=[("", "Can’t get item 4 of"), ("only 3 emails", "")],
        ) as capture:
            result = mail.get(number=5)
    assert result == "only 3 emails"
    assert capture.call_count == 2
    assert "repeat with i from 1 to 3" in capture.call_args_list[-1][0][0]
