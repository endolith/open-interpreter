import unittest

from interpreter.core.llm.llm import _TOOL_CALLING_INSTRUCTIONS


class TestToolCallingInstructions(unittest.TestCase):
    def test_instructions_do_not_duplicate_tool_schemas(self):
        self.assertNotIn("Two tools are available", _TOOL_CALLING_INSTRUCTIONS)
        self.assertNotIn("persistent REPL", _TOOL_CALLING_INSTRUCTIONS)
        self.assertNotIn("write`, `sed`", _TOOL_CALLING_INSTRUCTIONS)
        self.assertIn("JSON schema", _TOOL_CALLING_INSTRUCTIONS)
        self.assertIn("internal storage", _TOOL_CALLING_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
