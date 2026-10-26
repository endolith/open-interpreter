import platform
import unittest
from unittest import mock

from interpreter.core.toolbox.toolbox import Toolbox


class TestToolbox(unittest.TestCase):
    def setUp(self):
        self.toolbox = Toolbox(mock.Mock())

    def test_get_all_toolbox_tools_list(self):
        # Act
        tools_list = self.toolbox._get_all_toolbox_tools_list()

        # Assert: 12 core tools; macOS inserts mail, sms, calendar, contacts after clipboard (+4).
        if platform.system() == "Darwin":
            self.assertEqual(len(tools_list), 16)
        else:
            self.assertEqual(len(tools_list), 12)

    def test_get_all_toolbox_tools_list_order(self):
        """The declared subsystem order is preserved, with macOS extras inserted after clipboard."""
        expected = [
            self.toolbox.mouse,
            self.toolbox.keyboard,
            self.toolbox.display,
            self.toolbox.clipboard,
            self.toolbox.browser,
            self.toolbox.os,
            self.toolbox.web,
            self.toolbox.vision,
            self.toolbox.skills,
            self.toolbox.ai,
            self.toolbox.ai2,
            self.toolbox.files,
        ]
        if platform.system() == "Darwin":
            expected = (
                expected[:4]
                + [
                    self.toolbox.mail,
                    self.toolbox.sms,
                    self.toolbox.calendar,
                    self.toolbox.contacts,
                ]
                + expected[4:]
            )
        self.assertEqual(self.toolbox._get_all_toolbox_tools_list(), expected)

    def test_get_all_toolbox_tools_signature_and_description(self):
        # Act
        tools_description = self.toolbox._get_all_toolbox_tools_signature_and_description()

        # Assert: one string per exposed method (and optional module headers); count moves with the codebase.
        self.assertGreater(len(tools_description), 40)

    def test_signature_and_description_use_toolbox_prefix(self):
        """Every exposed method is described by its ``toolbox.<subsystem>`` name."""
        joined = "\n".join(
            self.toolbox._get_all_toolbox_tools_signature_and_description()
        )
        for subsystem in ("mouse", "keyboard", "display", "clipboard", "browser",
                          "os", "web", "vision", "skills", "ai", "files"):
            self.assertIn(f"toolbox.{subsystem}.", joined)


if __name__ == "__main__":
    testing = TestToolbox()
    testing.setUp()
    testing.test_get_all_toolbox_tools_signature_and_description()
