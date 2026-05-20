import platform

from .shell_preprocess import preprocess_shell
from .subprocess_language import SubprocessLanguage


class Cmd(SubprocessLanguage):
    file_extension = "bat"
    name = "cmd"
    execute_tool_hint = "Windows cmd.exe — set VAR=value, use %VAR%; not bash/PowerShell syntax"

    def __init__(self):
        super().__init__()
        if platform.system() != "Windows":
            raise RuntimeError("cmd language is only available on Windows")
        self.start_cmd = ["cmd.exe", "/K", "chcp 65001 >nul"]

    def preprocess_code(self, code):
        return preprocess_shell(code)

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line
