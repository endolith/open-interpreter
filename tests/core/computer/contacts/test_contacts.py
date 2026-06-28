from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.computer.contacts.contacts import Contacts


def test_get_phone_number_non_macos():
    contacts = Contacts(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.contacts.contacts.platform.system",
        return_value="Linux",
    ):
        assert contacts.get_phone_number("Alice") == "This method is only supported on MacOS"


def test_get_phone_number_success():
    contacts = Contacts(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.contacts.contacts.platform.system",
        return_value="Darwin",
    ):
        with mock.patch(
            "interpreter.core.computer.contacts.contacts.run_applescript_capture",
            return_value=("555-1234\n", ""),
        ):
            assert contacts.get_phone_number("Alice") == "555-1234"


def test_get_phone_number_suggests_similar_contacts():
    contacts = Contacts(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.contacts.contacts.platform.system",
        return_value="Darwin",
    ):
        with mock.patch(
            "interpreter.core.computer.contacts.contacts.run_applescript_capture",
            side_effect=[("", "Can't get person"), ("Alice Smith", "")],
        ):
            with mock.patch.object(
                contacts, "get_full_names_from_first_name", return_value="Alice Smith"
            ):
                with pytest.raises(Exception, match="similar contacts"):
                    contacts.get_phone_number("Ali")
