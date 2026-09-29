import os
import re

from .subprocess_language import SubprocessLanguage


class R(SubprocessLanguage):
    file_extension = "r"
    name = "R"

    def __init__(self):
        super().__init__()
        # --no-echo stops R echoing each line of input back on stdout. R echoes
        # even when stdin is a plain pipe, and the echoes interleave with real
        # output (markers, printed values, the end marker) rather than preceding
        # it, so they cannot be filtered afterwards by skipping a line count --
        # any fixed count either eats real output or leaks an echo. Suppressing
        # the echo at the source is what makes the output stream unambiguous.
        self.start_cmd = ["R", "-q", "--vanilla", "--no-echo"]

    def preprocess_code(self, code):
        """
        Add active line markers
        Wrap in a tryCatch for better error handling in R
        Add end of execution marker
        """

        active_line_enabled = (
            os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower() == "true"
        )

        lines = code.split("\n")
        processed_lines = []

        for i, line in enumerate(lines, 1):
            if active_line_enabled:
                processed_lines.append(f'cat("##active_line{i}##\\n");{line}')
            else:
                processed_lines.append(line)

        # Join lines to form the processed code
        processed_code = "\n".join(processed_lines)

        # Wrap in a tryCatch for error handling and add end of execution marker
        return f"""
tryCatch({{
{processed_code}
}}, error=function(e){{
    cat("##execution_error##\\n", conditionMessage(e), "\\n");
}})
cat("##end_of_execution##\\n");
"""

    def line_postprocessor(self, line):
        # A bare R continuation prompt carries no output of ours. With --no-echo
        # these should not appear at all, but dropping them is cheap insurance
        # for an R build that ignores the flag.
        if re.match(r"^(\s*>>>\s*|\s*\.\.\.\s*|\s*>\s*|\s*\+\s*|\s*)$", line):
            return None
        if "R version" in line:  # Startup message
            return None
        if line.strip().startswith('[1] "') and line.endswith(
            '"'
        ):  # For strings, trim quotation marks
            return line[5:-1].strip()
        if line.strip().startswith(
            "[1]"
        ):  # Normal R output prefix for non-string outputs
            return line[4:].strip()

        return line

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line or "##execution_error##" in line
