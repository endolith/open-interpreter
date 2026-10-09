import unittest
from pathlib import Path
from unittest import mock

from interpreter.core.computer.computer import COMPUTER_TOOL_SUBSYSTEMS, Computer

_COMPUTER_PKG = Path(__file__).resolve().parents[3] / "interpreter" / "core" / "computer"
_NON_TOOL_SUBPACKAGES = frozenset({"terminal", "utils", "__pycache__"})


class TestComputer(unittest.TestCase):
    def setUp(self):
        self.computer = Computer(mock.Mock())

    def test_get_all_computer_tools_list(self):
        """_get_all_computer_tools_list returns registered subsystems in order."""
        tools_list = self.computer._get_all_computer_tools_list()
        expected = [getattr(self.computer, name) for name in COMPUTER_TOOL_SUBSYSTEMS]
        self.assertEqual(tools_list, expected)

    def test_init_wires_all_registered_subsystems(self):
        """Every COMPUTER_TOOL_SUBSYSTEMS entry is attached on Computer in __init__."""
        for name in COMPUTER_TOOL_SUBSYSTEMS:
            self.assertTrue(
                hasattr(self.computer, name),
                f"Computer.__init__ must set self.{name} (see COMPUTER_TOOL_SUBSYSTEMS)",
            )

    def test_subpackage_dirs_match_tool_subsystems(self):
        """Each computer/ subpackage (except terminal, utils) is in COMPUTER_TOOL_SUBSYSTEMS."""
        subdirs = {
            path.name
            for path in _COMPUTER_PKG.iterdir()
            if path.is_dir() and path.name not in _NON_TOOL_SUBPACKAGES
        }
        self.assertEqual(set(COMPUTER_TOOL_SUBSYSTEMS), subdirs)

    def test_get_all_computer_tools_signature_and_description(self):
        """Each subsystem's dot-notation prefix appears in the tools description."""
        tools_description = self.computer._get_all_computer_tools_signature_and_description()
        joined = "\n".join(tools_description)

        self.assertGreater(len(joined), 64)
        for subsystem in COMPUTER_TOOL_SUBSYSTEMS:
            self.assertIn(f"computer.{subsystem}.", joined)


if __name__ == "__main__":
    testing = TestComputer()
    testing.setUp()
    testing.test_get_all_computer_tools_signature_and_description()
