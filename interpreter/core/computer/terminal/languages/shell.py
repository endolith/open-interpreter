import os
import platform
import re

from .subprocess_language import SubprocessLanguage


class Shell(SubprocessLanguage):
    file_extension = "sh"
    name = "Shell"
    aliases = ["bash", "sh", "zsh", "batch", "bat"]

    def __init__(
        self,
    ):
        super().__init__()

        # Determine the start command based on the platform
        if platform.system() == "Windows":
            self.start_cmd = ["cmd.exe"]
        else:
            self.start_cmd = [os.environ.get("SHELL", "bash")]

    def preprocess_code(self, code):
        return preprocess_shell(code)

    def line_postprocessor(self, line):
        return line

    def detect_active_line(self, line):
        # Program output is untrusted and can contain the marker text, so only
        # a complete "##active_line<digits>##" counts. Parsing anything that
        # merely contains "##active_line" raised ValueError here, which killed
        # the stdout reader thread and left run() waiting forever for the
        # end-of-execution marker.
        match = re.search(r"##active_line(\d+)##", line)
        if match:
            return int(match.group(1))
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line


def preprocess_shell(code):
    """
    Add active line markers
    Wrap in a try except (trap in shell)
    Add end of execution marker
    """

    # Add commands that tell us what the active line is
    # if it's multiline, just skip this. soon we should make it work with multiline
    if (
        not has_multiline_commands(code)
        and os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower()
        == "true"
    ):
        code = add_active_line_prints(code)

    # Add end command (we'll be listening for this so we know when it ends)
    code += '\necho "##end_of_execution##"'

    # WARNING: Do not add a wall-clock timeout for the end-of-execution marker.
    # Long-running shell jobs are valid. See PR #144 (closed) and issue #148.

    return code


def add_active_line_prints(code):
    """
    Add echo statements indicating line numbers to a shell string.
    """
    lines = code.split("\n")
    for index, line in enumerate(lines):
        # Insert the echo command before the actual line
        lines[index] = f'echo "##active_line{index + 1}##"\n{line}'
    return "\n".join(lines)


def spans_lines_inside_quoting(script_text):
    """
    Return True if a line boundary falls inside a quoted string or a heredoc.

    add_active_line_prints inserts an echo between every pair of physical
    lines, which is only valid where bash treats the line break as a command
    boundary. Inside a heredoc the echo becomes body text (silently corrupting
    a file the model is writing); inside a quoted string it becomes part of
    the string. Keyword matching cannot see either, so the text is scanned
    character by character for quoting state.
    """
    quote = None  # "'" or '"' while inside a quoted string
    index = 0
    length = len(script_text)

    while index < length:
        char = script_text[index]

        if quote == "'":
            # Single quotes are literal; only another single quote ends them.
            if char == "\n":
                return True
            if char == "'":
                quote = None
            index += 1
        elif quote == '"':
            if char == "\\" and index + 1 < length:
                if script_text[index + 1] == "\n":
                    # A line continuation inside quotes is still a line break
                    # an injected echo would land in the middle of.
                    return True
                index += 2  # Escaped character, e.g. \" or \$
                continue
            if char == "\n":
                return True
            if char == '"':
                quote = None
            index += 1
        elif char == "\\":
            index += 2  # Escaped character (a line continuation included)
        elif char == "#" and (index == 0 or script_text[index - 1] in " \t\n;&|()"):
            # Comments cannot open a quote or a heredoc, and an apostrophe in
            # one ("# don't") would otherwise look like an unclosed quote.
            newline = script_text.find("\n", index)
            index = length if newline == -1 else newline
        elif char in "'\"":
            quote = char
            index += 1
        elif script_text.startswith("<<", index):
            if script_text.startswith("<<<", index):
                index += 3  # Here-string: stays on one line, nothing to do
                continue
            # A heredoc body starts on the next line and has to stay contiguous
            # up to its delimiter.
            return True
        else:
            index += 1

    # An unterminated quote is broken code, but instrumenting it would only
    # make the error harder to read.
    return quote is not None


def has_multiline_commands(script_text):
    # Quoting and heredocs need a scan, not a per-line pattern match
    if spans_lines_inside_quoting(script_text):
        return True

    # Patterns that indicate a line continues
    continuation_patterns = [
        r"\\$",  # Line continuation character at the end of the line
        r"\|$",  # Pipe character at the end of the line indicating a pipeline continuation
        r"&&\s*$",  # Logical AND at the end of the line
        r"\|\|\s*$",  # Logical OR at the end of the line
        r"<\($",  # Start of process substitution
        r"\($",  # Start of subshell
        r"{\s*$",  # Start of a block
        r"\bif\b",  # Start of an if statement
        r"\bwhile\b",  # Start of a while loop
        r"\bfor\b",  # Start of a for loop
        r"\bcase\b",  # Start of a case statement (only patterns may follow)
        r"do\s*$",  # 'do' keyword for loops
        r"then\s*$",  # 'then' keyword for if statements
    ]

    # Check each line for multiline patterns
    for line in script_text.splitlines():
        if any(re.search(pattern, line.rstrip()) for pattern in continuation_patterns):
            return True

    return False
