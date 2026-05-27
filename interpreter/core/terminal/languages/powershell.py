import re

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
        # Strip PS prompt lines: "(base) PS C:\Users\...>" or "PS C:\...>"
        # These appear because OI feeds code to a persistent interactive REPL via
        # stdin, so PowerShell echoes each line back with its prompt.
        # Use re.match (anchored to start of line) so that legitimate output
        # containing "PS C:\" mid-line (e.g. "Path: PS C:\foo") is not dropped.
        if re.match(r"(\(.*?\) )?PS [A-Za-z]:\\", line):
            return None
        # Strip continuation-prompt echo lines (">> code" or bare ">>").
        stripped = line.rstrip("\r\n")
        if stripped == ">>" or stripped.startswith(">> "):
            return None
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
    # Write-Host to stdout keeps error output on one clean line.
    # Write-Error would include the full script-block context (the entire try{} wrapper)
    # which exposes OI's scaffolding and is not useful to the user or the LLM.
    return try_catch_code + code + '\n} catch {\n    Write-Host "Error: $($_.Exception.Message)"\n}\n'
