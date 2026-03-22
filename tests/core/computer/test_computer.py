import unittest
from unittest import mock
from interpreter.core.toolbox.toolbox import Toolbox

class TestToolbox(unittest.TestCase):
    def setUp(self):
        self.toolbox = Toolbox(mock.Mock())

    def test_get_all_toolbox_tools_list(self):
        # Act
        tools_list = self.toolbox._get_all_toolbox_tools_list()

        # Assert
        self.assertEqual(tools_list, expected)

    def test_get_all_toolbox_tools_signature_and_description(self):
        # Act
        tools_description = self.toolbox._get_all_toolbox_tools_signature_and_description()

        # Assert
        self.assertGreater(len(joined), 64)
        for subsystem in COMPUTER_TOOL_SUBSYSTEMS:
            self.assertIn(f"computer.{subsystem}.", joined)


if __name__ == "__main__":
    testing = TestToolbox()
    testing.setUp()
    testing.test_get_all_toolbox_tools_signature_and_description()