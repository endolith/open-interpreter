import re

from .subprocess_language import SubprocessLanguage

# Printed by preprocess_javascript as the first thing the program executes, so
# line_postprocessor can tell Node's echo of our own buffered input (prompts
# attached) from the program's output (never prompts).
_JS_FRAME_MARKER = "__oi_js_frame__"


class JavaScript(SubprocessLanguage):
    file_extension = "js"
    name = "JavaScript"

    def __init__(self):
        super().__init__()
        self.start_cmd = ["node", "-i"]
        self._js_frame_seen = False

    def preprocess_code(self, code):
        """Reset the output frame for a new run, then apply the JS wrapper.

        The frame marker has not been printed yet, so the next run's echo is
        prompt-stripped again.
        """
        self._js_frame_seen = False
        return preprocess_javascript(code)

    def line_postprocessor(self, line):
        """Drop REPL scaffolding, keep everything the program printed.

        Node's banner lines and undefined results never reach the user, and
        the continuation prompts of its own echoed input are stripped only
        before the frame marker; after it, leading bars are output.
        """
        # Node's interactive REPL outputs a billion things
        # So we clean it up:
        if "Welcome to Node.js" in line:
            return None
        if line.strip() in ["undefined", 'Type ".help" for more information.']:
            return None
        line = line.strip(". \n")
        # Remove trailing ">"s
        line = re.sub(r"^\s*(>\s*)+", "", line)
        # Node can print the marker behind its prompts (`> | | | | <marker>`),
        # so drop prefix prompts for the comparison only — never from a line
        # that is not the marker.
        if re.sub(r"^(?:\| )*", "", line).strip() == _JS_FRAME_MARKER:
            # Everything before this was Node echoing our own input; everything
            # after it is program output, which may legitimately start with
            # bars ("| | empty", a markdown table row).
            self._js_frame_seen = True
            return None
        # Node 24 renders a buffered multi-line command's continuation prompt as
        # "| " (readline's kMultilinePrompt), so the echoed input arrives as
        # "| | | | ...". Only strip while the frame has not started: once the
        # marker has printed, a leading "| " belongs to the program, and four
        # prompts plus two output bars would otherwise become "empty" instead
        # of "| | empty".
        if not self._js_frame_seen:
            line = re.sub(r"^(?:\| ){4,}", "", line)
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
console.log("{_JS_FRAME_MARKER}");
{code}
}} catch (e) {{
    console.log(e);
}}
console.log("##end_of_execution##");
"""

    return code
