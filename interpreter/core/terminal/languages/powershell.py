import os
import re

from .cwd_tracking import CwdTrackingMixin
from .resolve_powershell import powershell_startup_args, resolve_powershell_executable
from .subprocess_language import SubprocessLanguage

# pwsh brackets each stdin read with DECCKM sequences (ESC[?1h / ESC[?1l) when
# it echoes a prompt. On Linux these land *inside* the captured line, so a
# continuation echo can start with "\x1b[?1l>> " and a marker can arrive as
# "\x1b[?1l##oi_cwd##...", both of which the anchored checks below would miss.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# pwsh emits ESC[?1h right before the echoed copy of the typed line, which makes
# it an exact boundary between real output and OI's own input coming back.
_INPUT_ECHO_START = "\x1b[?1h"
# The prompt pwsh draws before that echo, when it lands after output that did
# not end in a newline ("xPS /path> ").
_PROMPT_TAIL_RE = re.compile(r"(?:\(.*?\) )?PS (?:[A-Za-z]:\\|/)[^>]*> ?\s*$")
_CONTINUATION_TAIL_RE = re.compile(r">> ?\s*$")


class PowerShell(CwdTrackingMixin, SubprocessLanguage):
    file_extension = "ps1"
    name = "powershell"
    execute_tool_hint = "PowerShell — $var = value; cmdlet syntax. Requires pwsh on Linux/Mac. One command per line for readability."
    cd_commands = ("cd", "Set-Location", "sl")
    cd_ignore_case = True
    # PowerShell has no `nul` device; redirecting to `nul` creates a literal file.
    null_device = "$null"
    shell_quote_style = "powershell"

    def __init__(self):
        CwdTrackingMixin.__init__(self)
        # Explicit, not super(): with this MRO super() would resolve to
        # CwdTrackingMixin again and skip SubprocessLanguage.__init__.
        SubprocessLanguage.__init__(self)
        self.start_cmd = [resolve_powershell_executable(), *powershell_startup_args()]

    def preprocess_code(self, code):
        code, _ = self._normalize_null_redirects(code)
        code = self._strip_redundant_cd(code)
        code = preprocess_powershell(code)
        end_marker = '\nWrite-Output "##end_of_execution##"'
        return self._insert_cwd_marker(code, end_marker)

    def _cwd_marker_echo(self):
        return '\nWrite-Output "##oi_cwd##$($PWD.Path)"'

    def line_postprocessor(self, line):
        # Strip terminal-control sequences before classifying the line; see
        # _ANSI_CSI_RE. Then drop echoed input (prompt + continuation) *before*
        # the marker filters in CwdTrackingMixin run. An echoed source line such
        # as `Write-Output "##oi_cwd##$($PWD.Path)"` contains the marker text,
        # so letting it reach the marker parser would record `$($PWD.Path)"` as
        # the working directory and leak the echoed tail as output.
        if _INPUT_ECHO_START in line:
            return self._output_before_input_echo(line)
        line = _ANSI_CSI_RE.sub("", line)
        if self._is_echo_line(line):
            return None
        return super().line_postprocessor(line)

    def _output_before_input_echo(self, line):
        # Everything from the echo marker onward is OI's own input coming back,
        # including any prompt text. Real output can precede it on the same line
        # when the previous command did not end with a newline (pwsh draws the
        # next prompt immediately after it), so keep that prefix and drop the
        # prompt tail plus the echo.
        head = line.split(_INPUT_ECHO_START, 1)[0]
        head = _ANSI_CSI_RE.sub("", head)
        head = _PROMPT_TAIL_RE.sub("", head)
        head = _CONTINUATION_TAIL_RE.sub("", head)
        if not head.strip():
            return None
        return head

    def _is_echo_line(self, line):
        # OI feeds code to a persistent interactive REPL through stdin, so pwsh
        # echoes each line back with its prompt. Match the prompt shape at the
        # start of the line only (a drive-letter path on Windows, an absolute
        # POSIX path on Unix, optionally behind a conda prefix), so real output
        # containing "PS ..." mid-line is kept.
        if re.match(r"(\(.*?\) )?PS ([A-Za-z]:\\|/)", line):
            return True
        # Continuation-prompt echo lines: ">> code" or a bare ">>".
        stripped = line.strip()
        return stripped == ">>" or stripped.startswith(">> ")

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line


def preprocess_powershell(code):
    """
    Add active line markers (when safe), wrap in try-catch, add end-of-execution marker.
    """
    # Inserting Write-Output between lines of a hash literal @{}, script block {},
    # pipeline continuation |, etc. causes parse errors.  Mirror bash's approach:
    # skip active-line markers whenever the code contains multiline constructs.
    if (
        not has_multiline_constructs(code)
        and os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower() == "true"
    ):
        code = add_active_line_prints(code)

    code = wrap_in_try_catch(code)

    code += '\nWrite-Output "##end_of_execution##"'

    return code


def has_multiline_constructs(code):
    """
    Return True if the code contains constructs that span multiple lines and
    would cause parse errors if Write-Output statements were inserted between lines.

    Covers: hash literals @{}, script blocks {}, pipeline continuations |,
    backtick line continuations `, here-strings @" / @'.
    Mirrors has_multiline_commands() in shell_preprocess.py.
    """
    patterns = [
        r"\{\s*$",      # line ending with { — script block, hash literal, if/for/try body
        r"\(\s*$",      # opening parenthesis at end of line
        r"\|\s*$",      # pipeline continuation
        r"`\s*$",       # backtick line continuation
        r"^@[\"']",     # here-string start (@" or @')
    ]
    for line in code.splitlines():
        if any(re.search(p, line.rstrip()) for p in patterns):
            return True
    return False


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
