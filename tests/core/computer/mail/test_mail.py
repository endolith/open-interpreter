from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.mail.mail import Mail


def test_mail_get_non_macos_returns_message():
    mail = Mail(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.mail.mail.platform.system", return_value="Linux"
    ):
        assert mail.get() == "This method is only supported on MacOS"


def test_mail_get_on_macos_runs_applescript():
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
