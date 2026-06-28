import unittest
from unittest import mock

from interpreter.core.computer.computer import Computer

from tests.conftest import COMPUTER_TOOL_SUBSYSTEMS


class TestComputer(unittest.TestCase):
    def setUp(self):
        self.computer = Computer(mock.Mock())

    def test_get_all_computer_tools_list(self):
        tools_list = self.computer._get_all_computer_tools_list()
        expected = [getattr(self.computer, name) for name in COMPUTER_TOOL_SUBSYSTEMS]
        self.assertEqual(tools_list, expected)

    def test_get_all_computer_tools_signature_and_description(self):
        tools_description = self.computer._get_all_computer_tools_signature_and_description()
        joined = "\n".join(tools_description)
        self.assertGreater(len(joined), 64)
        for subsystem in COMPUTER_TOOL_SUBSYSTEMS:
            self.assertIn(f"computer.{subsystem}.", joined)


if __name__ == "__main__":
    testing = TestComputer()
    testing.setUp()
    testing.test_get_all_computer_tools_signature_and_description()
