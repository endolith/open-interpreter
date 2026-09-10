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
        # Skip lines that execute nothing. Marking them is meaningless, and the
        # marker is an `echo`, so one sitting after the user's last real command
        # becomes the command that sets `$?`. Code ending in a newline splits to
        # a trailing empty line, which is how a failing command's exit status
        # used to be reported as 0. The index still counts skipped lines so the
        # numbers keep matching the user's source.
        if _runs_nothing(line):
            continue
        lines[index] = f'echo "##active_line{index + 1}##"\n{line}'
    return "\n".join(lines)


def _runs_nothing(line):
    """True for blank and comment-only lines.

    ``#`` is the comment character in bash; cmd uses ``rem``/``::`` instead, but
    skipping a ``#`` line there only costs a highlight for a line that would
    have failed anyway.
    """
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def has_multiline_commands(script_text):
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
        r"do\s*$",
        r"then\s*$",
    ]

    for line in script_text.splitlines():
        if any(re.search(pattern, line.rstrip()) for pattern in continuation_patterns):
            return True

    return False
