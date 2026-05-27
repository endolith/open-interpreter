import unittest

from interpreter import interpreter
from interpreter.core.utils.system_debug_info import _llm_prompt_sections_for_info


class TestSystemDebugInfo(unittest.TestCase):
    def test_info_includes_execute_language_modes_in_tool_calling_mode(self):
        interpreter.llm.supports_functions = True
        sections = dict(_llm_prompt_sections_for_info(interpreter))
        self.assertIn("System Message (tool-calling mode)", sections)
        self.assertIn("Execute tool", sections)
        self.assertIn("persistent REPL", sections["Execute tool"])
        self.assertIn("Edit tool", sections)
        self.assertIn("write", sections["Edit tool"])

    def test_info_includes_execution_instructions_in_text_mode(self):
        interpreter.llm.supports_functions = False
        sections = dict(_llm_prompt_sections_for_info(interpreter))
        self.assertIn("System Message (text / markdown mode)", sections)
        self.assertIn(interpreter.llm.execution_instructions, sections["System Message (text / markdown mode)"])


if __name__ == "__main__":
    unittest.main()
