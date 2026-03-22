import unittest
from pathlib import Path
from unittest import mock
from interpreter.core.toolbox.toolbox import Toolbox

class TestToolbox(unittest.TestCase):
    def setUp(self):
        self.toolbox = Toolbox(mock.Mock())

    def test_get_all_toolbox_tools_list(self):
        # Act
        tools_list = self.toolbox._get_all_toolbox_tools_list()

    def test_wired_subsystems_match_tools_list_in_order(self):
        """Eagerly wired subsystems appear in the tools list in __init__ order."""
        tools = self.computer._get_all_computer_tools_list()
        expected = [
            getattr(self.computer, name)
            for name in _wired_subsystem_names(self.computer)
        ]
        self.assertEqual(tools, expected)

    def test_property_subsystems_are_in_tools_list(self):
        """A lazy @property subsystem must be listed, or explicitly exempted."""
        tools = self.computer._get_all_computer_tools_list()
        for name in _public_property_names():
            self.assertIn(
                getattr(self.computer, name),
                tools,
                f"Computer.{name} is a public tool property but is missing from "
                "_get_all_computer_tools_list(); add it or exempt it in "
                "_NON_TOOL_PROPERTIES",
            )

    def test_get_all_toolbox_tools_signature_and_description(self):
        # Act
        tools_description = self.toolbox._get_all_toolbox_tools_signature_and_description()

        # Assert
        self.assertGreater(len(joined), 64)
        for name in _filesystem_subsystem_names() | _public_property_names():
            self.assertIn(f"computer.{name}.", joined)


if __name__ == "__main__":
    testing = TestToolbox()
    testing.setUp()
    testing.test_get_all_toolbox_tools_signature_and_description()