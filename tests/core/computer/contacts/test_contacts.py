from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.computer.contacts.contacts import Contacts


DARWIN = "interpreter.core.computer.contacts.contacts.platform.system"
CAPTURE = "interpreter.core.computer.contacts.contacts.run_applescript_capture"


def _darwin():
    return mock.patch(DARWIN, return_value="Darwin")


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
            side_effect=[("", "Can't get person"), ("Alice Smith", "")],
        ):
            with mock.patch.object(contacts, "get_full_names_from_first_name", return_value="Alice Smith"):
                with pytest.raises(Exception, match="similar contacts"):
                    contacts.get_phone_number("Ali")


def test_get_phone_number_raises_when_no_similar_contacts():
    """With no similar contacts, the lookup raises plain "Contact not found"."""
    contacts = Contacts(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "Can’t get person")):
            with mock.patch.object(
                contacts,
                "get_full_names_from_first_name",
                return_value="No contacts found",
            ):
                with pytest.raises(Exception, match="Contact not found"):
                    contacts.get_phone_number("Zzz")


def test_get_phone_number_empty_output_triggers_similar_search():
    """Empty stdout counts as a miss even without an error message."""
    contacts = Contacts(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "")):
            with mock.patch.object(
                contacts,
                "get_full_names_from_first_name",
                return_value="",
            ):
                with pytest.raises(Exception, match="Contact not found"):
                    contacts.get_phone_number("Zzz")


def test_get_email_address_success():
    """On macOS, get_email_address() returns the trimmed lookup result."""
    contacts = Contacts(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("a@example.com\n", "")):
            assert contacts.get_email_address("Alice") == "a@example.com"


def test_get_email_address_suggests_similar_contacts():
    """A missed email lookup returns suggestions instead of raising.

    Unlike the phone variant, which raises, the email variant returns the
    suggestion text — pinned as written so a future merge of the two paths
    shows up here.
    """
    contacts = Contacts(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "Can’t get person")):
            with mock.patch.object(
                contacts,
                "get_full_names_from_first_name",
                return_value="Alice Smith",
            ):
                result = contacts.get_email_address("Ali")
    assert "similar contacts" in result
    assert "Alice Smith" in result


@pytest.mark.xfail(reason='get_email_address() compares against "No contacts found" exactly (issue #411), so the real lookup result "No contacts found." misses and falls into the similar-contacts suggestion')
def test_get_email_address_no_contacts_found():
    """With no similar contacts, the email lookup returns a plain message.

    The real empty lookup returns "No contacts found." (with the period —
    see get_full_names_from_first_name()). Once get_email_address()
    recognises that spelling, this mock follows the no-contacts path.
    Today the exact comparison misses and the result is instead offered
    back as a similar-contact suggestion.
    """
    contacts = Contacts(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("", "Can’t get person")):
            with mock.patch.object(
                contacts,
                "get_full_names_from_first_name",
                return_value="No contacts found.",
            ):
                assert contacts.get_email_address("Zzz") == "No contacts found"


def test_get_email_address_non_macos():
    """get_email_address() refuses off macOS instead of running AppleScript."""
    contacts = Contacts(computer=SimpleNamespace())
    with mock.patch(DARWIN, return_value="Linux"):
        assert contacts.get_email_address("Alice") == ("This method is only supported on MacOS")


def test_get_full_names_returns_matches_or_fallback():
    """Matches pass through; empty output becomes "No contacts found"."""
    contacts = Contacts(computer=SimpleNamespace())
    with _darwin():
        with mock.patch(CAPTURE, return_value=("Alice Smith, Alice Jones", "")):
            assert contacts.get_full_names_from_first_name("Alice") == ("Alice Smith, Alice Jones")
        with mock.patch(CAPTURE, return_value=("", "")):
            assert contacts.get_full_names_from_first_name("Zzz") == "No contacts found."


def test_get_full_names_non_macos():
    """get_full_names_from_first_name() refuses off macOS."""
    contacts = Contacts(computer=SimpleNamespace())
    with mock.patch(DARWIN, return_value="Linux"):
        assert contacts.get_full_names_from_first_name("Alice") == ("This method is only supported on MacOS")
