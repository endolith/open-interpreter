import re
import subprocess

from .subprocess_language import SubprocessLanguage


class Augeas(SubprocessLanguage):
    file_extension = "aug"
    name = "augeas"
    execute_tool_hint = (
        "augtool commands (set/get/load/save/print). Each block runs a fresh "
        "augtool; call save to persist changes to disk."
    )

    def __init__(self):
        super().__init__()
        self.start_cmd = ["augtool"]

    def preprocess_code(self, code):
        return code.strip() + "\n" if code.strip() else "\n"

    def run(self, code):
        """Run one augtool block to completion and yield its output.

        Driving a persistent augtool over pipes cannot work reliably: it only
        emits its prompt on a tty, and with a pipe it processes one buffered
        line at a time, so prompt-based end detection never fires and the turn
        hangs. A one-shot process exits at EOF, which is a reliable terminator;
        the trade-off is that state does not carry between blocks (use `save`
        to persist changes).
        """
        try:
            result = subprocess.run(
                self.start_cmd,
                input=self.preprocess_code(code),
                text=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            yield {
                "type": "console",
                "format": "output",
                "content": "augtool timed out after 120 seconds.",
            }
            return
        except Exception as e:
            yield {
                "type": "console",
                "format": "output",
                "content": f"augtool failed to start: {e}",
            }
            return

        output = (result.stdout or "") + (result.stderr or "")
        for line in output.splitlines():
            if line.strip():
                yield {
                    "type": "console",
                    "format": "output",
                    "content": line + "\n",
                }

    def line_postprocessor(self, line):
        if re.match(r"^\s*augtool>\s*$", line):
            return None
        return line if line.strip() else None

    def detect_end_of_execution(self, line):
        return False
