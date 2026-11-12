import re

from .subprocess_language import SubprocessLanguage


class JavaScript(SubprocessLanguage):
    file_extension = "js"
    name = "JavaScript"

    def __init__(self):
        super().__init__()
        self.start_cmd = ["node", "-i"]

    def preprocess_code(self, code):
        return preprocess_javascript(code)

    def line_postprocessor(self, line):
        # Node's interactive REPL outputs a billion things
        # So we clean it up:
        if "Welcome to Node.js" in line:
            return None
        if line.strip() in ["undefined", 'Type ".help" for more information.']:
            return None
        line = line.strip(". \n")
        # Strip the REPL prompt sequence, including the continuation prompts
        # for multi-line input: "> ", "> > ", "... ... ". Without the "..." arm
        # Node echoes a buffered multi-line command as "... ... ... " glued to
        # the result line, which the model then reads as output.
        line = re.sub(r"^\s*(?:(?:>\s*)|(?:\.\.\.\s*))+", "", line)
        # Node 24 renders a buffered multi-line command's continuation prompt as
        # "| " (readline's kMultilinePrompt) and emits the run on the result
        # line. Our try/catch wrapper is at least seven lines, so a real run is
        # long; requiring four leaves ordinary output such as a markdown table
        # row ("| a | b |", or "| | empty") untouched.
        #
        # The prompt token and a row's first cell delimiter are the same "| ",
        # so a table row that directly follows the prompts loses its leading
        # delimiter along with them ("| | | | | | | | a | b |" -> "a | b |").
        # Where the prompts stop is not recoverable from the line alone, so put
        # one delimiter back when what remains still looks like a row: it ends
        # with a delimiter and has another one inside. Rows with several leading
        # empty cells stay ambiguous, as does a value that merely ends in "|".
        line, prompt_run = re.subn(r"^(?:\| ){4,}", "", line)
        if prompt_run and line.endswith("|") and line.count("|") >= 2:
            line = "| " + line
        return line

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line


def preprocess_javascript(code):
    """
    Add active line markers
    Wrap in a try catch
    Add end of execution marker
    """

    # Detect if nothing in the code is multiline. (This is waaaay to false-positive-y but it works)
    nothing_multiline = not any(char in code for char in ["{", "}", "[", "]"])

    if nothing_multiline:
        # Split code into lines
        lines = code.split("\n")
        processed_lines = []
        for i, line in enumerate(lines, 1):
            # Add active line print
            processed_lines.append(f'console.log("##active_line{i}##");')
            processed_lines.append(line)

        # Join lines to form the processed code
        code = "\n".join(processed_lines)

    # Wrap in a try-catch and add end of execution marker
    code = f"""
try {{
{code}
}} catch (e) {{
    console.log(e);
}}
console.log("##end_of_execution##");
"""

    return code
