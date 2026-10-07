from types import SimpleNamespace
from unittest import mock

import subprocess

import pytest

from interpreter.core.computer.mail.mail import Mail


DARWIN = "interpreter.core.computer.mail.mail.platform.system"
CAPTURE = "interpreter.core.computer.mail.mail.run_applescript_capture"
RUN = "interpreter.core.computer.mail.mail.run_applescript"


def _darwin():
    return mock.patch(DARWIN, return_value="Darwin")


def test_mail_get_non_macos_returns_message():
    """Mail.get() returns an unsupported-platform message on non-macOS systems."""
    mail = Mail(computer=SimpleNamespace())
    with mock.patch("interpreter.core.computer.mail.mail.platform.system", return_value="Linux"):
        assert mail.get() == "This method is only supported on MacOS"


def test_mail_get_on_macos_runs_applescript():
    """On macOS, Mail.get() runs AppleScript limited to the requested message count."""
    mail = Mail(computer=SimpleNamespace())
    with mock.patch("interpreter.core.computer.mail.mail.platform.system", return_value="Darwin"):
        with mock.patch(
            "interpreter.core.computer.mail.mail.run_applescript_capture",
            return_value=("inbox data", ""),
        ) as capture:
            result = mail.get(number=2)
    capture.assert_called_once()
    script = capture.call_args[0][0]
    assert "repeat with i from 1 to 2" in script
    assert result == "inbox data"


def test_mail_get_caps_large_requests_with_a_warning():
    """Asking for 60 emails caps the script at 50 and prefixes the warning.

    The message text says "10 emails" while the code caps at 50 — pinned
    exactly as written, since either the message or the cap looks wrong and
    changing one without the other would hide which.
    """
    mail = Mail(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("inbox data", "")) as capture:
            result = mail.get(number=60)
    script = capture.call_args[0][0]
    assert "repeat with i from 1 to 50" in script
    assert result.startswith("This method is limited to 10 emails, returning the first 10: ")
    assert result.endswith("inbox data")


def test_mail_get_retries_with_fewer_when_mailbox_is_short():
    """ "Can't get item N" retries once with N-1 instead of failing outright."""
    mail = Mail(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(
            CAPTURE,
            side_effect=[
                ("", "Can’t get item 5 of every message"),
                ("short inbox", ""),
            ],
        ) as capture:
            assert mail.get(number=5) == "short inbox"
    assert capture.call_count == 2
    assert "repeat with i from 1 to 4" in capture.call_args[0][0]


def test_mail_get_gives_up_after_three_short_retries():
    """Three short-mailbox retries exhaust the loop and return None."""
    mail = Mail(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "Can’t get item 5 of every message")) as capture:
            assert mail.get(number=5) is None
    assert capture.call_count == 3


def test_mail_get_breaks_when_item_count_unparseable():
    """ "Can't get item" without a number breaks out after one attempt."""
    mail = Mail(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "Can’t get item of every message")) as capture:
            assert mail.get(number=5) is None
    capture.assert_called_once()


def test_mail_get_unparseable_short_error_never_stops_asking():
    """An unparseable short-mailbox error neither retries nor breaks.

    With empty stdout and no "item N" to parse, the loop body makes no
    progress and `retries` never increments, so get() keeps re-sending the
    same script. Pinned by exhausting the mock: after the scripted responses
    run out, the mock raises StopIteration instead of get() returning, and
    every call asked for the same count. Tracked as #409.
    """
    mail = Mail(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "Can’t get messages")) as capture:
            capture.side_effect = [("", "Can’t get messages")] * 3
            with pytest.raises(StopIteration):
                mail.get(number=5)
    assert capture.call_count == 4
    for call in capture.call_args_list:
        assert "repeat with i from 1 to 5" in call[0][0]


def test_mail_get_unread_filters_by_read_status():
    """unread=True adds the read-status filter to the AppleScript."""
    mail = Mail(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("unread data", "")) as capture:
            assert mail.get(unread=True) == "unread data"
    assert "whose read status is false" in capture.call_args[0][0]


def test_mail_send_strips_newlines_and_reports_success():
    """Newlines are stripped from the recipient so one address stays one line."""
    mail = Mail(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(RUN) as run:
            assert mail.send("a@x.com\nb@y.com", "hi", "body") == ("Email sent to a@x.comb@y.com")
    script = run.call_args[0][0]
    assert 'address:"a@x.comb@y.com"' in script
    assert "delay" not in script


def test_mail_send_includes_attachments_and_delay():
    """Attachments become AppleScript clauses and size the post-send delay."""
    mail = Mail(computer=SimpleNamespace())
    with _darwin():
        with mock.patch.object(mail, "calculate_upload_delay", return_value=7.5):
            with mock.patch(RUN) as run:
                with mock.patch.object(
                    mail,
                    "format_path_for_applescript",
                    side_effect=lambda p: f"POSIX file {p!r}",
                ):
                    result = mail.send("a@x.com", "hi", "body", attachments=["/tmp/f.txt"])
    assert result == "Email sent to a@x.com"
    script = run.call_args[0][0]
    assert "POSIX file '/tmp/f.txt'" in script
    assert "delay 7.5" in script


def test_mail_send_failure_returns_failure_message():
    """An osascript failure surfaces as a failure string, not an exception."""
    mail = Mail(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(RUN, side_effect=subprocess.CalledProcessError(1, "osascript")):
            assert mail.send("a@x.com", "hi", "body") == "Failed to send email"


def test_mail_send_non_macos_returns_message():
    """Mail.send() refuses off macOS instead of running AppleScript."""
    mail = Mail(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.mail.mail.platform.system",
        return_value="Linux",
    ):
        with mock.patch(RUN) as run:
            assert mail.send("a@x.com", "s", "b") == ("This method is only supported on MacOS")
    run.assert_not_called()


def test_mail_unread_count_caps_at_fifty_or_more():
    """50+ unread is reported as a bucket, fewer as the exact count."""
    mail = Mail(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(RUN, return_value="73"):
            assert mail.unread_count() == "50 or more"
        with mock.patch(RUN, return_value="7\n"):
            assert mail.unread_count() == 7


def test_mail_unread_count_failure_returns_zero():
    """An osascript failure yields 0, so callers get a number either way."""
    mail = Mail(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(RUN, side_effect=subprocess.CalledProcessError(1, "osascript")):
            assert mail.unread_count() == 0


def test_mail_unread_count_non_macos_returns_message():
    """unread_count() refuses off macOS instead of running AppleScript."""
    mail = Mail(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.mail.mail.platform.system",
        return_value="Linux",
    ):
        assert mail.unread_count() == "This method is only supported on MacOS"


def test_calculate_upload_delay_scales_with_file_size(tmp_path):
    """~2 MB at the assumed 1 MB/s plus the 1 s buffer rounds to one decimal."""
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (2 * 1024 * 1024))
    mail = Mail(computer=SimpleNamespace())
    assert mail.calculate_upload_delay([str(big)]) == 3.0
    tiny = tmp_path / "tiny.bin"
    tiny.write_bytes(b"x")
    # A near-empty file still waits out the 1 s buffer (the 0.2 s floor is
    # unreachable beside it, so the minimum observable delay is 1.0).
    assert mail.calculate_upload_delay([str(tiny)]) == 1.0


def test_calculate_upload_delay_missing_file_returns_default():
    """An unreadable attachment falls back to the 5 s default, not an error."""
    mail = Mail(computer=SimpleNamespace())
    assert mail.calculate_upload_delay(["/no/such/file.bin"]) == 5


def test_format_path_for_applescript_escapes_and_quotes():
    """Backslashes, quotes, and braces are escaped inside a POSIX file quote."""
    mail = Mail(computer=SimpleNamespace())
    assert mail.format_path_for_applescript('a\\b"c{d}e') == ('POSIX file "a\\\\b\\"c\\{d\\}e"')
