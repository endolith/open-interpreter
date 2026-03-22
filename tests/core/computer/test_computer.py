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

        # Assert: 13 core tools; macOS inserts mail, sms, calendar, contacts after clipboard (+4).
        if platform.system() == "Darwin":
            self.assertEqual(len(tools_list), 17)
        else:
            self.assertEqual(len(tools_list), 13)

    def test_get_all_toolbox_tools_signature_and_description(self):
        # Act
        tools_description = self.toolbox._get_all_toolbox_tools_signature_and_description()

        # Assert: one string per exposed method (and optional module headers); count moves with the codebase.
        self.assertGreater(len(tools_description), 40)

if __name__ == "__main__":
    testing = TestToolbox()
    testing.setUp()
    testing.test_get_all_toolbox_tools_signature_and_description()