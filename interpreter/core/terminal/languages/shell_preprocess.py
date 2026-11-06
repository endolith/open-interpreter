import os
import re


def preprocess_shell(code):
    """
    Add active line markers, end-of-execution marker (echo works in bash and cmd).
    """
    if (
        not has_multiline_commands(code)
        and os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower()
        == "true"
    ):
        code = add_active_line_prints(code)

    code += '\necho "##end_of_execution##"'

    return code


def add_active_line_prints(code):
    lines = code.split("\n")
    for index, line in enumerate(lines):
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

    continuation_patterns = [
        r"\\$",
        r"\|$",
        r"&&\s*$",
        r"\|\|\s*$",
        r"<\($",
        r"\($",
        r"{\s*$",
        r"\bif\b",
        r"\bwhile\b",
        r"\bfor\b",
        r"\bcase\b",  # Start of a case statement (only patterns may follow)
        r"do\s*$",
        r"then\s*$",
    ]

    for line in script_text.splitlines():
        if any(re.search(pattern, line.rstrip()) for pattern in continuation_patterns):
            return True

    return False
