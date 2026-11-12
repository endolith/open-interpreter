import re

from .subprocess_language import SubprocessLanguage


class JavaScript(SubprocessLanguage):
    file_extension = "js"
    name = "JavaScript"

    def __init__(self):
        super().__init__()
        self.start_cmd = ["node", "-i"]
        # Number of continuation prompts Node will print for the code most
        # recently passed to preprocess_code; 0 until a run has been prepared.
        self._continuation_prompts = 0

    def preprocess_code(self, code):
        processed = preprocess_javascript(code)
        # Node buffers the whole wrapper before executing it and prints one
        # continuation prompt per buffered line, less a constant three. Counting
        # them lets line_postprocessor strip exactly the prompts instead of an
        # arbitrary run length, so a result line that itself begins with "| "
        # (a markdown table row) keeps its own delimiters. The offset is
        # verified on Node 22.22.1 and 24.22.3 for single-line, multi-line,
        # array, blank-line, form-feed and trailing-newline inputs.
        # Node's readline treats only CR, LF and CRLF as line breaks; Python's
        # splitlines() also splits on form feed, vertical tab and Unicode line
        # separators, which would overcount the prompts for those characters.
        lines = re.split(r"\r\n|\r|\n", processed)
        if lines[-1] == "":
            lines.pop()
        self._continuation_prompts = max(0, len(lines) - 3)
        return processed

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
        #
        # A ">" prefix marks the line Node glued its continuation prompts to
        # (a line that begins with "..." has already lost its dots in the strip
        # above). Remember the prefix before the regex consumes it: later
        # output lines must not be treated as prompt lines even when they
        # begin with "| ", because that is a table row's own delimiter.
        prompt = re.match(r"^\s*(?:>\s*)+", line)
        if prompt is None:
            return line
        line = re.sub(r"^\s*(?:>\s*)+", "", line)
        # Node 22 prints "..." as its input continuation prompt, and a result
        # line that itself starts with dots ("...waiting") is glued right
        # after the last prompt. Strip exactly the number of prompts
        # preprocess_code recorded so the result keeps its own dots; the
        # wholesale strip below is only a fallback for direct calls, which
        # have no count.
        if re.match(r"^(?:\.\.\.\s*)+", line):
            if self._continuation_prompts:
                line, prompt_run = re.subn(
                    "^" + "... " * self._continuation_prompts, "", line, count=1
                )
                if not prompt_run:
                    line = re.sub(r"^(?:\.\.\.\s*)+", "", line)
            else:
                line = re.sub(r"^(?:\.\.\.\s*)+", "", line)
            # Node 22 never prints "| " prompts, so the remaining "| " run
            # belongs to the result line (e.g. a table row) and must not be
            # stripped.
            return line
        # Node 24 renders a buffered multi-line command's continuation prompt as
        # "| " (readline's kMultilinePrompt) and emits the run on the result
        # line. Strip exactly the number of prompts preprocess_code counted for
        # this run. A run-length test alone cannot tell the prompts from a
        # table row's own leading delimiters ("| a | b |", or "| | | a | b |"
        # with empty leading cells), so it consumed the row's delimiter too.
        # Fall back to the old run-length heuristic when the count is unknown
        # (line_postprocessor called directly) or Node printed a different
        # number of prompts than expected.
        if self._continuation_prompts:
            line, prompt_run = re.subn(
                r"^(?:\| ){%d}" % self._continuation_prompts, "", line
            )
            if prompt_run:
                return line
        # Our try/catch wrapper is at least seven lines, so a real prompt run is
        # long; requiring four leaves ordinary output such as a markdown table
        # row untouched.
        line, prompt_run = re.subn(r"^(?:\| ){4,}", "", line)
        if prompt_run and line.endswith("|") and line.count("|") >= 2:
            # The prompt and a row's first cell delimiter are the same "| ",
            # so the run-length strip can eat one of the row's delimiters.
            # Put one back when what remains still looks like a row: it ends
            # with a delimiter and has another one inside. Rows with several
            # leading empty cells stay ambiguous on this fallback path, which
            # is why the counted path above is preferred.
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
