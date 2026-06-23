"""
This is the template Open Interpreter profile.

A starting point for creating a new profile.

Learn about all the available settings - https://github.com/endolith/open-interpreter/blob/main/docs/settings/all-settings.mdx

"""

# Import the interpreter
from interpreter import interpreter

# You can import other libraries too
from datetime import date

# You can set variables
today = date.today()

# LLM Settings
interpreter.llm.model = "groq/llama-3.3-70b-versatile"
interpreter.llm.context_window = 110000
interpreter.llm.max_tokens = 4096
interpreter.llm.api_base = "https://api.example.com"
interpreter.llm.api_key = "your_api_key_here"
interpreter.llm.supports_functions = False
interpreter.llm.supports_vision = False


# Interpreter Settings
interpreter.offline = False
interpreter.loop = True
interpreter.auto_run = False

# Toggle OS Mode - https://github.com/endolith/open-interpreter/blob/main/docs/guides/os-mode.mdx
interpreter.os = False

# Import Computer API - https://github.com/endolith/open-interpreter/blob/main/docs/code-execution/computer-api.mdx
interpreter.computer.import_computer_api = True


# Set Custom Instructions to improve your Interpreter's performance at a given task
interpreter.custom_instructions = f"""
    Today's date is {today}.
    """
