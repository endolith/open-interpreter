import re

from .subprocess_language import SubprocessLanguage


class JavaScript(SubprocessLanguage):
    file_extension = "js"
    name = "JavaScript"

    def __init__(self):
        super().__init__()
        self.start_cmd = ["node", "-i"]
        # True while Node is still echoing the buffered source of the current
        # run, when a run of "| " tokens is a readline continuation prompt.
        self._echoing_input = True

    def preprocess_code(self, code):
        # Each run echoes its input again, so the prompt phase restarts.
        self._echoing_input = True
        return preprocess_javascript(code)

    def line_postprocessor(self, line):
        # Node's interactive REPL outputs a billion things, so clean it up.
        if "Welcome to Node.js" in line:
            return None
        if line.strip() in ["undefined", 'Type ".help" for more information.']:
            return None
        # Strip the REPL prompt sequence, including continuation prompts for
        # multi-line input: "> ", "> > ", "... ...". These otherwise leak into
        # the output. Only the prompt tokens are removed; the rest of the line,
        # including the trailing newline, is preserved. Matching only horizontal
        # whitespace here is deliberate: a `\s` class also consumes the newline
        # on prompt-only lines, which would reintroduce joined output.
        line = re.sub(r"^[ \t]*(?:(?:>[ \t]*)|(?:\.\.\.[ \t]*))+", "", line)
        # Node 24 renders a buffered multi-line command's continuation prompt as
        # "| " (readline's kMultilinePrompt) and echoes the run on the result
        # line. Those prompts only precede the echoed source; after the wrapper's
        # active-line marker the program's own output begins. Requiring four
        # tokens keeps short markdown rows ("| a | b |") intact, and the phase
        # check keeps a literal "| | | | value |" from being eaten.
        # stderr never carries the REPL's own echo, so a program writing a
        # markdown row to stderr must keep its leading pipes even while stdout
        # is still echoing the source.
        if self._echoing_input and not self.current_stream_is_error:
            line = re.sub(r"^(?:\| ){4,}", "", line)
            # The wrapper prints an active-line marker before any program output,
            # so the marker ends the echoed-input phase. Output after it is the
            # program's own and may legitimately start with four "| " tokens (a
            # table row such as "| | | | value |"), which must not be deleted.
            if "##active_line" in line or "##end_of_execution##" in line:
                self._echoing_input = False
        # Keep the trailing newline: Terminal.run(stream=False) concatenates
        # consecutive output chunks, so stripping it joins separate lines.
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
