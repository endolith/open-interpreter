import unittest

from interpreter.core.default_system_message import default_system_message
from interpreter.core.llm.run_function_calling_llm import function_schema
from interpreter.core.llm.run_tool_calling_llm import tool_schema
from interpreter.core.terminal.base_language import (
    format_execute_language_description,
)
from interpreter.core.terminal.languages.bash import Bash
from interpreter.core.terminal.languages.html import HTML
from interpreter.core.terminal.languages.java import Java
from interpreter.core.terminal.languages.python import Python


class TestExecutionStatePrompt(unittest.TestCase):
    def test_tool_schema_requires_incremental_persistent_calls(self):
        """Verify the current execute schema tells models to continue live sessions rather than restart them."""
        code_description = tool_schema["function"]["parameters"]["properties"]["code"][
            "description"
        ]
        self.assertIn("next missing operation", code_description)
        self.assertIn("Jupyter/IPython kernel", code_description)
        self.assertIn("stateless/display-only", code_description)

    def test_legacy_function_schema_matches_tool_schema_contract(self):
        """Verify older function-calling models receive the same persistent-session contract."""
        code_description = function_schema["parameters"]["properties"]["code"][
            "description"
        ]
        self.assertIn("next missing operation", code_description)
        self.assertIn("Jupyter/IPython kernel", code_description)
        self.assertIn("stateless/display-only", code_description)

    def test_language_description_covers_all_execution_modes(self):
        """Verify the dynamic language list distinguishes persistent, stateless, and display-only execution."""
        description = format_execute_language_description(
            [Python, Bash, Java, HTML]
        )
        self.assertIn("persistent REPL", description)
        self.assertIn("next missing operation", description)
        self.assertIn("Jupyter/IPython kernel", description)
        self.assertIn("complete, self-contained block", description)
        self.assertIn("complete display source", description)

    def test_system_message_models_define_once_and_call_later(self):
        """Verify the system prompt contains a reusable helper pattern rather than repeated setup."""
        self.assertIn("continue the live session; do not restart it", default_system_message)
        self.assertIn("Repeated setup after unchanged state is incorrect", default_system_message)
        self.assertIn("def mean(values)", default_system_message)
        self.assertIn("mean(second_values)", default_system_message)
        self.assertIn("Stay in the live session", default_system_message)


if __name__ == "__main__":
    unittest.main()
