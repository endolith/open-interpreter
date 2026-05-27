from .resolve_powershell import powershell_startup_args, resolve_powershell_executable
from .subprocess_language import SubprocessLanguage


class PowerShell(SubprocessLanguage):
    file_extension = "ps1"
    name = "powershell"
    execute_tool_hint = "PowerShell — $var = value; cmdlet syntax. Requires pwsh on Linux/Mac."

    def __init__(self):
        super().__init__()
        self.start_cmd = [resolve_powershell_executable(), *powershell_startup_args()]

    def preprocess_code(self, code):
        return preprocess_powershell(code)

    def line_postprocessor(self, line):
        return line

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line


def preprocess_powershell(code):
    """
    Add active line markers
    Wrap in try-catch block
    Add end of execution marker
    """
    code = add_active_line_prints(code)

    code = wrap_in_try_catch(code)

    code += '\nWrite-Output "##end_of_execution##"'

    return code


def add_active_line_prints(code):
    lines = code.split("\n")
    for index, line in enumerate(lines):
        lines[index] = f'Write-Output "##active_line{index + 1}##"\n{line}'
    return "\n".join(lines)


def wrap_in_try_catch(code):
    try_catch_code = """
try {
    $ErrorActionPreference = "Stop"
"""
    return try_catch_code + code + "\n} catch {\n    Write-Error $_\n}\n"
