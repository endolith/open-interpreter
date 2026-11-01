import os
import re
from pathlib import Path
from .subprocess_language import SubprocessLanguage


class Ruby(SubprocessLanguage):
    file_extension = "rb"
    name = "Ruby"

    def __init__(self):
        super().__init__()
        self.start_cmd = ["irb"] 

    def preprocess_code(self, code):
        """
        Add active line markers
        Wrap in a tryCatch for better error handling 
        Add end of execution marker
        """

        active_line_enabled = (
            os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower() == "true"
        )

        lines = code.split("\n")
        processed_lines = []

        for i, line in enumerate(lines, 1):
            if active_line_enabled:
                processed_lines.append(f'puts "##active_line{i}##"')
            processed_lines.append(line)
        # Join lines to form the processed code
        processed_code = "\n".join(processed_lines)

        # Wrap in a tryCatch for error handling and add end of execution marker
        processed_code = f"""
begin
  {processed_code}
rescue Exception => e
  puts "##execution_error##\\n" + e.message
ensure
  puts "##end_of_execution##\\n"
end
"""
        self.code_line_count = len(processed_code.split("\n"))
        #print(processed_code)
        return processed_code

    def line_postprocessor(self, line):
        # If the line count attribute is set and non-zero, decrement and skip the line
        if hasattr(self, "code_line_count") and self.code_line_count > 0:
            self.code_line_count -= 1
            return None
        if "nil" in line:
           return None
        return line

    def detect_end_of_execution(self, line):
        # A syntax error is raised by the parser before the wrapper's ensure
        # can run, so the marker never prints and the run would wait forever.
        # irb reports it on its own traceback; treat that as the end too.
        # An incomplete block (an unterminated `def`) prints nothing at all,
        # so there is nothing to detect here. That parser-state hang is covered
        # by the periodic wait-or-kill check-in in #401, with #270 as the
        # automatic idle-timeout backstop.
        return (
            "##end_of_execution##" in line
            or "##execution_error##" in line
            or "(SyntaxError)" in line
        )