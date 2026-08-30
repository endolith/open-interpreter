import platform

from .cwd_tracking import CwdTrackingMixin
from .shell_preprocess import preprocess_shell
from .subprocess_language import SubprocessLanguage


class Cmd(CwdTrackingMixin, SubprocessLanguage):
    file_extension = "bat"
    name = "cmd"
    execute_tool_hint = "Windows cmd.exe — set VAR=value, use %VAR%; not bash/PowerShell syntax"
    # `cd /d X` is cmd's change-drive form; `;` is not a cmd separator.
    cd_option_prefixes = ("/d",)
    cd_chain_operators = ("&&", "&")

    def __init__(self):
        CwdTrackingMixin.__init__(self)
        # Explicit, not super(): with this MRO super() would resolve to
        # CwdTrackingMixin again and skip SubprocessLanguage.__init__.
        SubprocessLanguage.__init__(self)
        if platform.system() != "Windows":
            raise RuntimeError("cmd language is only available on Windows")
        self.start_cmd = ["cmd.exe", "/K", "chcp 65001 >nul"]

    def preprocess_code(self, code):
        code = self._strip_redundant_cd(code)
        code = preprocess_shell(code)
        end_marker = '\necho "##end_of_execution##"'
        return self._insert_cwd_marker(code, end_marker)

    def _cwd_marker_echo(self):
        # `@` suppresses cmd echoing this command back; `%CD%` is the current dir.
        return "\n@echo ##oi_pwd##%CD%"

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line
