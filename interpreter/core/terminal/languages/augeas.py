import re

from .subprocess_language import SubprocessLanguage


class Augeas(SubprocessLanguage):
    file_extension = "aug"
    name = "augeas"
    execute_tool_hint = (
        "augtool commands (set/get/load/save/print); session persists — "
        "call save to write changes back to disk"
    )

    def __init__(self):
        super().__init__()
        self.start_cmd = ["augtool"]
        self._in_block = False

    def preprocess_code(self, code):
        return code.strip() + "\n" if code.strip() else "\n"

    def run(self, code):
        self._in_block = True
        try:
            yield from super().run(code)
        finally:
            self._in_block = False

    def line_postprocessor(self, line):
        if re.match(r"^\s*augtool>\s*$", line):
            return None
        return line if line.strip() else None

    def detect_end_of_execution(self, line):
        if not self._in_block:
            return False
        return bool(re.match(r"^\s*augtool>\s*$", line))
