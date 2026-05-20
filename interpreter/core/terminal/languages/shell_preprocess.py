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
