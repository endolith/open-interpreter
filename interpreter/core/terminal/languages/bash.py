import re

from .cwd_tracking import CwdTrackingMixin
from .resolve_bash import resolve_bash_executable
from .shell_preprocess import preprocess_shell
from .subprocess_language import SubprocessLanguage


class Bash(CwdTrackingMixin, SubprocessLanguage):
    file_extension = "sh"
    name = "bash"
    execute_tool_hint = "GNU bash — export VAR=value; always bash, never the login shell (fish/zsh)"
    # POSIX escaping: `cd My\ Documents` is one path with a space.
    cd_unescape_backslashes = True

    def __init__(self):
        CwdTrackingMixin.__init__(self)
        # Call SubprocessLanguage.__init__ explicitly: with this MRO, super()
        # would resolve to CwdTrackingMixin again and skip it entirely.
        SubprocessLanguage.__init__(self)
        self.start_cmd = [resolve_bash_executable()]

    # Matches the end-of-execution marker with the exit status appended to it.
    _EXIT_CODE_RE = re.compile(r"##end_of_execution##(\d+)")

    def preprocess_code(self, code):
        code = self._strip_redundant_cd(code)
        code = preprocess_shell(code)
        # Save $? the instant the user's code finishes. The cwd echo that
        # _insert_cwd_marker adds below runs before the end marker, so reading
        # $? in the marker itself would report that echo's status (always 0)
        # rather than the user's.
        code = code.replace(
            '\necho "##end_of_execution##"',
            '\n__oi_exit_code=$?\necho "##end_of_execution##$__oi_exit_code"',
        )
        end_marker = '\necho "##end_of_execution##$__oi_exit_code"'
        return self._insert_cwd_marker(code, end_marker)

    def _cwd_marker_echo(self):
        return '\necho "##oi_pwd##$PWD"'

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line

    def detect_exit_code(self, line):
        match = self._EXIT_CODE_RE.search(line)
        return int(match.group(1)) if match else None
