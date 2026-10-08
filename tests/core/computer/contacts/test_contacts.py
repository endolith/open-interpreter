from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.computer.contacts.contacts import Contacts


def test_get_phone_number_non_macos():
    """get_phone_number() returns an unsupported-platform message on non-macOS systems."""
    contacts = Contacts(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.contacts.contacts.platform.system",
        return_value="Linux",
    ):
        assert contacts.get_phone_number("Alice") == "This method is only supported on MacOS"


def test_get_phone_number_success():
    """On macOS, get_phone_number() returns the trimmed AppleScript lookup result."""
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
    """When lookup fails, get_phone_number() raises with similar contact suggestions."""
    contacts = Contacts(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.contacts.contacts.platform.system",
        return_value="Darwin",
    ):
        with mock.patch(
            "interpreter.core.computer.contacts.contacts.run_applescript_capture",
            side_effect=[("", "Can’t get person"), ("Alice Smith", "")],
        ):
            with mock.patch.object(contacts, "get_full_names_from_first_name", return_value="Alice Smith"):
                with pytest.raises(Exception, match="similar contacts"):
                    contacts.get_phone_number("Ali")


def test_get_email_address_returns_address_on_success():
    """On macOS, get_email_address() returns the trimmed AppleScript lookup result."""
    contacts = Contacts(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.contacts.contacts.platform.system",
        return_value="Darwin",
    ):
        with mock.patch(
            "interpreter.core.computer.contacts.contacts.run_applescript_capture",
            return_value=("alice@example.com\n", ""),
        ):
            assert contacts.get_email_address("Alice") == "alice@example.com"


def test_get_email_address_reports_no_contacts_found():
    """A failed lookup whose fallback finds nothing returns the "No contacts found" sentinel (issue #411).

    The sentinel is what get_full_names_from_first_name actually returns, so it
    has to be matched as a substring: an exact equality test let the message
    fall through to the "perhaps one of these similar contacts" branch, which
    then suggested a contact list that did not exist.
    """
    contacts = Contacts(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.contacts.contacts.platform.system",
        return_value="Darwin",
    ):
        with mock.patch(
            "interpreter.core.computer.contacts.contacts.run_applescript_capture",
            return_value=("", "Can’t get person"),
        ):
            with mock.patch.object(contacts, "get_full_names_from_first_name", return_value="No contacts found"):
                assert contacts.get_email_address("Ali") == "No contacts found"


def test_get_email_address_suggests_similar_contacts():
    """A failed lookup that finds similar names returns the model-friendly hint."""
    contacts = Contacts(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.contacts.contacts.platform.system",
        return_value="Darwin",
    ):
        with mock.patch(
            "interpreter.core.computer.contacts.contacts.run_applescript_capture",
            return_value=("", "Can’t get person"),
        ):
            with mock.patch.object(
                contacts,
                "get_full_names_from_first_name",
                return_value="Alice Smith, Alice Jones",
            ):
                result = contacts.get_email_address("Ali")

    assert "similar contacts" in result
    assert "Alice Smith, Alice Jones" in result
