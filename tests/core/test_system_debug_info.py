import unittest

from interpreter import interpreter
from interpreter.core.utils.system_debug_info import _llm_prompt_sections_for_info


class TestSystemDebugInfo(unittest.TestCase):
    def test_info_includes_tools_json_in_tool_calling_mode(self):
        interpreter.llm.supports_functions = True
        sections = dict(_llm_prompt_sections_for_info(interpreter))
        self.assertIn("System message (`messages[0]`)", sections)
        tools_section = sections["Tools (`request.tools` JSON)"]
        self.assertIn("```json", tools_section)
        self.assertIn('"name": "execute"', tools_section)
        self.assertIn("persistent REPL", tools_section)
        self.assertIn('"name": "edit"', tools_section)
        self.assertIn('"write"', tools_section)

    def test_info_includes_execution_instructions_in_text_mode(self):
        interpreter.llm.supports_functions = False
        sections = dict(_llm_prompt_sections_for_info(interpreter))
        self.assertIn("System Message (text / markdown mode)", sections)
        self.assertIn(interpreter.llm.execution_instructions, sections["System Message (text / markdown mode)"])


if __name__ == "__main__":
    unittest.main()
