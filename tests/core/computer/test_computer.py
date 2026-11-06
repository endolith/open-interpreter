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


from pathlib import Path

_TOOLBOX_PKG = (
    Path(__file__).resolve().parents[3] / "interpreter" / "core" / "toolbox"
)
_NON_TOOL_DIRS = frozenset({"utils", "__pycache__"})
_NON_TOOL_MODULES = frozenset({"__init__.py", "toolbox.py"})
_NON_TOOL_PROPERTIES = frozenset({"languages", "system_message", "import_computer_api"})
_MAC_ONLY = frozenset({"mail", "sms", "calendar", "contacts"})


def _filesystem_subsystem_names():
    """Subsystem names implied by toolbox/ subpackages and single-file modules."""
    names = {
        path.name
        for path in _TOOLBOX_PKG.iterdir()
        if path.is_dir() and path.name not in _NON_TOOL_DIRS
    }
    names.update(
        path.stem
        for path in _TOOLBOX_PKG.glob("*.py")
        if path.name not in _NON_TOOL_MODULES
    )
    return names


def _public_property_names():
    """Public @property descriptors on Toolbox, excluding known non-tool properties."""
    return {
        name
        for name, value in vars(Toolbox).items()
        if not name.startswith("_")
        and isinstance(value, property)
        and name not in _NON_TOOL_PROPERTIES
    }


def _wired_subsystem_names(toolbox):
    """Eagerly assigned Toolbox attributes whose value is an interpreter.toolbox object."""
    return [
        name
        for name, value in vars(toolbox).items()
        if not name.startswith("_")
        and name != "interpreter"
        and type(value).__module__.startswith("interpreter.core.toolbox.")
    ]


class TestToolboxWiring(unittest.TestCase):
    def setUp(self):
        self.toolbox = Toolbox(mock.Mock())

    def test_filesystem_subsystems_are_wired(self):
        """Every toolbox/ subpackage or module is attached during Toolbox.__init__."""
        for name in _filesystem_subsystem_names():
            self.assertTrue(
                hasattr(self.toolbox, name),
                f"Toolbox.__init__ must set self.{name} for toolbox/{name}",
            )

    def test_wired_subsystems_match_tools_list_in_order(self):
        """Eagerly wired subsystems appear in the tools list in __init__ order."""
        tools = self.toolbox._get_all_toolbox_tools_list()
        eager_names = [
            name
            for name in _wired_subsystem_names(self.toolbox)
            if name not in _MAC_ONLY or platform.system() == "Darwin"
        ]
        tool_names = [tool.__class__.__name__.lower() for tool in tools]
        self.assertEqual(
            [name for name in tool_names if name in eager_names], eager_names
        )

    def test_property_subsystems_are_in_tools_list(self):
        """A lazy @property subsystem must be listed, or explicitly exempted."""
        tools = self.toolbox._get_all_toolbox_tools_list()
        for name in _public_property_names():
            self.assertIn(
                getattr(self.toolbox, name),
                tools,
                f"Toolbox.{name} is a public tool property but is missing from "
                "_get_all_toolbox_tools_list(); add it or exempt it in "
                "_NON_TOOL_PROPERTIES",
            )

    def test_get_all_toolbox_tools_signature_and_description(self):
        """Each discovered subsystem has a toolbox.<name>. prompt signature."""
        joined = "\n".join(
            self.toolbox._get_all_toolbox_tools_signature_and_description()
        )
        for tool in self.toolbox._get_all_toolbox_tools_list():
            name = tool.__class__.__name__.lower()
            self.assertIn(f"toolbox.{name}.", joined)


if __name__ == "__main__":
    testing = TestToolboxWiring()
    testing.setUp()
    testing.test_get_all_toolbox_tools_signature_and_description()
