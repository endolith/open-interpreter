import unittest
from pathlib import Path
from unittest import mock

from interpreter.core.computer.computer import Computer

_COMPUTER_PKG = (
    Path(__file__).resolve().parents[3] / "interpreter" / "core" / "computer"
)
_NON_TOOL_DIRS = frozenset({"terminal", "utils", "__pycache__"})
_NON_TOOL_MODULES = frozenset({"__init__.py", "computer.py"})
_NON_TOOL_PROPERTIES = frozenset({"languages"})


def _filesystem_subsystem_names():
    """Subsystem names implied by computer/ subpackages and single-file modules."""
    names = {
        path.name
        for path in _COMPUTER_PKG.iterdir()
        if path.is_dir() and path.name not in _NON_TOOL_DIRS
    }
    names.update(
        path.stem
        for path in _COMPUTER_PKG.glob("*.py")
        if path.name not in _NON_TOOL_MODULES
    )
    return names


def _public_property_names():
    """Public @property descriptors on Computer, excluding known non-tool properties."""
    return {
        name
        for name, value in vars(Computer).items()
        if not name.startswith("_")
        and isinstance(value, property)
        and name not in _NON_TOOL_PROPERTIES
    }


def _wired_subsystem_names(computer):
    """Eagerly assigned Computer attributes whose value is an interpreter.computer object."""
    return [
        name
        for name, value in vars(computer).items()
        if not name.startswith("_")
        and name != "terminal"
        and type(value).__module__.startswith("interpreter.core.computer.")
    ]


class TestComputer(unittest.TestCase):
    def setUp(self):
        self.computer = Computer(mock.Mock())

    def test_filesystem_subsystems_are_wired(self):
        """Every computer/ subpackage or module is attached during Computer.__init__."""
        for name in _filesystem_subsystem_names():
            self.assertTrue(
                hasattr(self.computer, name),
                f"Computer.__init__ must set self.{name} for computer/{name}",
            )

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

    def test_get_all_computer_tools_signature_and_description(self):
        """Each discovered subsystem has a computer.<name>. prompt signature."""
        joined = "\n".join(
            self.computer._get_all_computer_tools_signature_and_description()
        )
        self.assertGreater(len(joined), 64)
        for name in _filesystem_subsystem_names() | _public_property_names():
            self.assertIn(f"computer.{name}.", joined)


if __name__ == "__main__":
    testing = TestComputer()
    testing.setUp()
    testing.test_get_all_computer_tools_signature_and_description()
