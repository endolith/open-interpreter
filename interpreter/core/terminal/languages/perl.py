from pathlib import Path

from .subprocess_language import SubprocessLanguage


class Perl(SubprocessLanguage):
    file_extension = "pl"
    name = "perl"
    execute_tool_hint = (
        "full Perl scripts — open/read/write files, use %state across blocks; "
        "not for edit-tool one-liners (those use edit language perl)"
    )

    def __init__(self):
        super().__init__()
        repl = Path(__file__).with_name("perl_repl.pl")
        self.start_cmd = ["perl", str(repl)]

    def preprocess_code(self, code):
        return code.rstrip() + "\n__OI_END__\n"

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line or "##execution_error##" in line
