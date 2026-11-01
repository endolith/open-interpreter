import types
import unittest

from interpreter.core.default_system_message import default_system_message
from interpreter.core.utils.assemble_system_message import assemble_system_message
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

    def test_system_message_describes_only_the_active_execution_mode(self):
        """Tool mode must not mention fences, and fence mode must not mention tool calls."""
        tool = assemble_system_message(_prompt_interpreter(True))
        self.assertIn("call to the execute tool", tool)
        self.assertIn("[execute:", tool)
        self.assertNotIn("[code:", tool)
        self.assertNotIn("fenced code block", tool)

        fence = assemble_system_message(_prompt_interpreter(False))
        self.assertIn("fenced code block", fence)
        self.assertIn("[code:", fence)
        self.assertNotIn("[execute:", fence)
        self.assertNotIn("call to the execute tool", fence)

    def test_tool_instructions_show_a_positive_call_example(self):
        """Tool mode gets a concrete call, not just prohibitions about message code."""
        from interpreter.core.llm.llm import _TOOL_CALLING_INSTRUCTIONS

        self.assertIn("call `execute`", _TOOL_CALLING_INSTRUCTIONS)
        self.assertIn(
            'execute({"language": "python", "code": "print(1)"})',
            _TOOL_CALLING_INSTRUCTIONS,
        )
        self.assertIn("never executed", _TOOL_CALLING_INSTRUCTIONS)

    def test_custom_system_message_is_not_replaced(self):
        """A profile's own system_message stays exactly as written."""
        interpreter = _prompt_interpreter(True)
        interpreter.system_message = "CUSTOM PROMPT"

        self.assertEqual(assemble_system_message(interpreter), "CUSTOM PROMPT")

    def test_system_message_models_define_once_and_call_later(self):
        """Verify the system prompt contains a reusable helper pattern rather than repeated setup."""
        self.assertIn("continue the live session; do not restart it", default_system_message)
        self.assertIn("Repeated setup after unchanged state is incorrect", default_system_message)
        self.assertIn("def mean(values)", default_system_message)
        self.assertIn("mean(second_values)", default_system_message)
        self.assertIn("Stay in the live session", default_system_message)


def _prompt_interpreter(supports_functions):
    """A minimal interpreter shaped for assemble_system_message (no LLM, kernel, or toolbox)."""
    return types.SimpleNamespace(
        system_message=default_system_message,
        llm=types.SimpleNamespace(supports_functions=supports_functions),
        terminal=types.SimpleNamespace(languages=[]),
        custom_instructions="",
        toolbox=types.SimpleNamespace(
            save_skills=False, import_toolbox_api=False, system_message=""
        ),
        verbose=False,
        _server_request_system=None,
        _launch_cwd=None,
        _launch_time=None,
    )


if __name__ == "__main__":
    unittest.main()
