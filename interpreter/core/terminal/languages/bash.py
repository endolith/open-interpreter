from .cwd_tracking import CwdTrackingMixin
from .resolve_bash import resolve_bash_executable
from .shell_preprocess import preprocess_shell
from .subprocess_language import SubprocessLanguage


class Bash(CwdTrackingMixin, SubprocessLanguage):
    file_extension = "sh"
    name = "bash"
    execute_tool_hint = "GNU bash — export VAR=value; always bash, never the login shell (fish/zsh)"

    def __init__(self):
        CwdTrackingMixin.__init__(self)
        # Call SubprocessLanguage.__init__ explicitly: with this MRO, super()
        # would resolve to CwdTrackingMixin again and skip it entirely.
        SubprocessLanguage.__init__(self)
        self.start_cmd = [resolve_bash_executable()]

    def preprocess_code(self, code):
        code = self._strip_redundant_cd(code)
        code = preprocess_shell(code)
        end_marker = '\necho "##end_of_execution##"'
        return self._insert_cwd_marker(code, end_marker)

    def _cwd_marker_echo(self):
        return '\necho "##oi_pwd##$PWD"'

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line
