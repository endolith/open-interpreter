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
        # The wrapper prints this before a multi-line block, which receives no
        # per-line markers. It ends the echoed-input phase and is wrapper
        # plumbing rather than output, so drop it. Requiring stdout keeps a
        # program that prints the literal text on stderr from ending the phase.
        if not self.current_stream_is_error and "##start_of_execution##" in line:
            self._echoing_input = False
            return None
        # Wrapper markers are plumbing, not output. They arrive as statements,
        # so their echo carries prompt tokens even when a multi-line block has
        # already ended the phase through its start marker. Strip the tokens
        # here regardless of the phase; the marker itself is kept so the reader
        # can still emit the active-line event and see the end of execution.
        if not self.current_stream_is_error and (
            "##active_line" in line or "##end_of_execution##" in line
        ):
            self._echoing_input = False
            line = re.sub(r"^[ \t]*(?:(?:>[ \t]*)|(?:\.\.\.[ \t]*))+", "", line)
            line = re.sub(r"^(?:\| ){4,}", "", line)
            return line
        # Strip the REPL's prompt sequence, including continuation prompts for
        # multi-line input: "> ", "> > ", "... ...". These otherwise leak into
        # the output. Only the prompt tokens are removed; the rest of the line,
        # including the trailing newline, is preserved. Matching only horizontal
        # whitespace here is deliberate: a `\s` class also consumes the newline
        # on prompt-only lines, which would reintroduce joined output.
        #
        # Node 24 also renders a buffered multi-line command's continuation prompt
        # as "| " (readline's kMultilinePrompt) and echoes the run on the result
        # line. Requiring four tokens for that one keeps short markdown rows
        # ("| a | b |") intact.
        #
        # Both forms only precede the echoed source. After the wrapper's
        # active-line marker the program's own output begins and must keep
        # literal prefixes, and stderr never carries the REPL's own echo, so the
        # cleanup runs only while echoing stdout.
        if self._echoing_input and not self.current_stream_is_error:
            line = re.sub(r"^[ \t]*(?:(?:>[ \t]*)|(?:\.\.\.[ \t]*))+", "", line)
            line = re.sub(r"^(?:\| ){4,}", "", line)
        elif not self.current_stream_is_error and re.fullmatch(
            r"[ \t]*(?:(?:>[ \t]*)|(?:\.\.\.[ \t]*))+", line
        ):
            # Readline writes the prompt before reading input, so a live prompt
            # arrives as a final ">"-only line with no separator when the run
            # ends. Program output that merely looks like a prompt ends with a
            # newline (console.log does), is not matched here, and is kept. stderr
            # never carries a prompt, so a program writing ">" there keeps it too.
            return None
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
    else:
        # Multi-line code gets no per-line markers: a console.log statement
        # between its lines would split brackets and object literals. Emit one
        # marker before the block instead, so the reader can tell that the
        # echoed source (when the REPL prints it) is over and that a later
        # "> " prefix is the program's own output.
        code = f'console.log("##start_of_execution##");\n{code}'

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
