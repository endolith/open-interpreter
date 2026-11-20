import re

from .subprocess_language import SubprocessLanguage


class JavaScript(SubprocessLanguage):
    file_extension = "js"
    name = "JavaScript"

    # Node's REPL prefixes echoed source with prompt tokens. They only precede
    # input the REPL echoed itself, never the program's own output.
    _PROMPT_PREFIX = re.compile(r"^[ \t]*(?:(?:>[ \t]*)|(?:\.\.\.[ \t]*))+")
    _PIPE_PROMPT_PREFIX = re.compile(r"^(?:\| ){4,}")
    # Node renders a buffered multi-line command's continuation prompt as "| "
    # (readline's kMultilinePrompt) and, with piped output, mixes it with the
    # regular "> " and "... " tokens on one line. Marker recognition needs its
    # own fragment: a single "| " token suffices there because the whole-line,
    # stdout, and echo-phase guards already exclude program text, while the
    # general cleanup keeps its four-token threshold so markdown rows survive.
    _MARKER_PROMPT_RUN = r"(?:>[ \t]*)|(?:\.\.\.[ \t]*)|(?:\| )"
    _MARKER_PROMPT_PREFIX = re.compile(r"^[ \t]*(?:" + _MARKER_PROMPT_RUN + r")+")
    # A wrapper start-marker statement is exactly a prompt run plus the marker.
    # Requiring at least one prompt token and matching the whole line keeps a
    # program that prints the marker text (`console.log("##start_of_execution##")`
    # or with a literal prefix) from ending the echoed-input phase and losing
    # its own output.
    _START_MARKER_LINE = re.compile(
        r"[ \t]*(?:" + _MARKER_PROMPT_RUN + r")+##start_of_execution##\r?\n?"
    )

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
        # program that prints the literal text on stderr from ending the phase,
        # and matching the whole line keeps a program that prints the marker
        # text on stdout from being mistaken for the wrapper's own statement.
        if (
            not self.current_stream_is_error
            and self._echoing_input
            and self._START_MARKER_LINE.fullmatch(line)
        ):
            self._echoing_input = False
            return None
        # Wrapper markers are plumbing, not output. They arrive as statements,
        # so their echo carries prompt tokens even when a multi-line block has
        # already ended the phase through its start marker. Strip only the
        # tokens that immediately precede a marker; the marker itself is kept so
        # the reader can still emit the active-line event and see the end of
        # execution, and output sharing the line keeps its literal prefix.
        if not self.current_stream_is_error and (
            "##active_line" in line or "##end_of_execution##" in line
        ):
            was_echoing_input = self._echoing_input
            self._echoing_input = False
            # Only the echoed statement carries REPL prompts. Once execution
            # has started, text in front of an active-line marker is the
            # program's own output: `process.stdout.write("> ")` leaves it on
            # the next statement's marker line, and the reader must keep it.
            # The end marker is printed after the program's last newline, so it
            # never shares a line with output: its prompt run is always echo,
            # and leaving it would surface as a stray ">" chunk.
            if not was_echoing_input and "##active_line" in line:
                return line
            return self._strip_marker_prompts(line)
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
            line = self._PROMPT_PREFIX.sub("", line)
            line = self._PIPE_PROMPT_PREFIX.sub("", line)
        elif not self.current_stream_is_error and self._PROMPT_PREFIX.fullmatch(line):
            # Readline writes the prompt before reading input, so a live prompt
            # arrives as a final ">"-only line with no separator when the run
            # ends. Program output that merely looks like a prompt ends with a
            # newline (console.log does), is not matched here, and is kept. stderr
            # never carries a prompt, so a program writing ">" there keeps it too.
            return None
        # Keep the trailing newline: Terminal.run(stream=False) concatenates
        # consecutive output chunks, so stripping it joins separate lines.
        return line

    def _strip_marker_prompts(self, line):
        """Strip REPL prompt tokens that immediately precede a wrapper marker.

        Node echoes the marker statements too, so their line can carry prompt
        tokens, and with piped output it can mix "> ", "... " and "| " forms on
        one line. Remove the whole leading run at once, then keep the line only
        when the text left behind begins with a marker: text the program wrote
        keeps its literal prefix even when the marker shares its line.
        """
        stripped = self._MARKER_PROMPT_PREFIX.sub("", line)
        if stripped != line and stripped.startswith(
            ("##active_line", "##end_of_execution##", "##start_of_execution##")
        ):
            return stripped
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
        # User code goes in its own block so declarations such as
        # `const console = { log() {} }` stay out of the marker's lexical
        # scope; sharing the try block would leave the marker in their
        # temporal dead zone and fail the run before any user statement ran.
        code = f'console.log("##start_of_execution##");\n{{\n{code}\n}}'

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
