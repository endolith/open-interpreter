from .resolve_bash import resolve_bash_executable
from .shell_preprocess import preprocess_shell
from .subprocess_language import SubprocessLanguage


class Bash(SubprocessLanguage):
    file_extension = "sh"
    name = "bash"
    execute_tool_hint = "GNU bash — export VAR=value; always bash, never the login shell (fish/zsh)"

    def __init__(self):
        super().__init__()
        self.start_cmd = [resolve_bash_executable(), "-i"]

    def preprocess_code(self, code):
        return preprocess_shell(code)

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line
