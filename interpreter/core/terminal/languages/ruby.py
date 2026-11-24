import ntpath
import os
import re
import shutil
import subprocess
from pathlib import Path
from .subprocess_language import SubprocessLanguage


def _irb_start_cmd():
    r"""Build an irb launch command CreateProcess accepts.

    Popen(["irb"]) relies on PATH lookup of the launcher script, which POSIX
    does itself. Windows resolves executables through CreateProcess, which
    only accepts .exe images — a Ruby launcher (.bat/.cmd) fails with
    WinError 2 — so the resolved launcher runs through cmd.exe instead. When
    irb cannot be resolved the bare name is kept, so a missing Ruby still
    fails the same way it did before.

    The launcher is handed over in /s mode wrapped in its own quote pair:
    cmd strips exactly the outer pair and leaves the path quoted, so it stays
    one command. list2cmdline only quotes on whitespace, so without this a
    path such as ``C:\Ruby&Tools\bin\irb.cmd`` would split on ``&`` and one
    with parentheses would break where they appear.
    """
    launcher = shutil.which("irb")
    if launcher and os.name == "nt" and launcher.lower().endswith((".bat", ".cmd")):
        return [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/s",
            "/c",
            f'""{launcher}""',
        ]
    return ["irb"]


def _taskkill_path():
    """Absolute taskkill.exe path, so teardown does not depend on PATH.

    Built with ntpath rather than os.path so the result is the Windows path
    it would be on the host that uses it, and the tests can assert it from
    any platform.
    """
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return ntpath.join(root, "System32", "taskkill.exe")

# irb prints this once per session when it starts echoing piped input. It is a
# session banner, not part of any block's echo, so it must not consume a line
# from the echo window.
_IRB_BANNER_RE = re.compile(r"Switch to(?: non)? inspect mode\.\s*$")


class Ruby(SubprocessLanguage):
    file_extension = "rb"
    name = "Ruby"

    def __init__(self):
        super().__init__()
        self.start_cmd = _irb_start_cmd()
        # True only for the cmd.exe-wrapped form. With the plain ["irb"] form
        # the process is irb itself, so the base terminate() already stops it.
        self._launched_through_cmd = self.start_cmd != ["irb"]

    def terminate(self):
        """Stop irb, taking the cmd.exe wrapper's tree down on Windows.

        cmd.exe does not propagate termination to the program it launched, so
        terminating the wrapper alone leaves irb alive holding the pipes.
        taskkill /T ends the whole tree; the base implementation then reaps the
        wrapper and closes the streams.
        """
        if self._launched_through_cmd and self.process and os.name == "nt":
            try:
                subprocess.run(
                    [
                        _taskkill_path(),
                        "/T",
                        "/F",
                        "/PID",
                        str(self.process.pid),
                    ],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass
        super().terminate()

    def preprocess_code(self, code):
        """
        Add active line markers
        Wrap in a tryCatch for better error handling 
        Add end of execution marker
        """

        active_line_enabled = (
            os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower() == "true"
        )

        lines = code.split("\n")
        processed_lines = []

        for i, line in enumerate(lines, 1):
            if active_line_enabled:
                processed_lines.append(f'puts "##active_line{i}##"')
            processed_lines.append(line)
        # Join lines to form the processed code
        processed_code = "\n".join(processed_lines)

        # Wrap in a tryCatch for error handling and add end of execution marker
        processed_code = f"""
begin
  {processed_code}
rescue Exception => e
  puts "##execution_error##\\n" + e.message
ensure
  puts "##end_of_execution##\\n"
end
"""
        # The window must span exactly the lines irb echoes back. split("\n")
        # adds a phantom trailing element for the final newline, so the window
        # closed one line too late on every block after the first -- once the
        # one-time session banner was no longer there to spend that extra line,
        # it ate the block's first real output line.
        self.code_line_count = len(processed_code.splitlines())
        #print(processed_code)
        return processed_code

    def line_postprocessor(self, line):
        # Control markers close the echoed-input window. irb echoes the
        # submitted block and adds its own banner, so the line count drifts;
        # letting the window swallow an exact marker loses the completion
        # signal and the next run waits forever. The echoed source lines are
        # not exact markers (they read `puts "##end_of_execution##\n"`), so
        # this exemption cannot mistake an echo for the real marker.
        stripped = line.strip()
        if self.code_line_count > 0 and _IRB_BANNER_RE.fullmatch(stripped):
            # One-time session banner: consume it without spending a source line,
            # or the echoed block's last line would leak as output.
            return None
        if (
            re.fullmatch(r"##active_line\d+##", stripped)
            or stripped.startswith("##execution_error##")
            or stripped == "##end_of_execution##"
        ):
            self.code_line_count = 0
            return line
        # If the line count attribute is set and non-zero, decrement and skip the line
        if hasattr(self, "code_line_count") and self.code_line_count > 0:
            self.code_line_count -= 1
            return None
        # irb prints a bare `nil` for a nil result. Match only that value so
        # strings such as "vanilla" or "nilpotent" survive as real output.
        if stripped in ("nil", "=> nil"):
            return None
        return line

    def detect_end_of_execution(self, line):
        # A syntax error is raised by the parser before the wrapper's ensure
        # can run, so the marker never prints and the run would wait forever.
        # irb reports it on its own traceback; treat that as the end too.
        # An incomplete block (an unterminated `def`) prints nothing at all,
        # so there is nothing to detect here. That parser-state hang is covered
        # by the periodic wait-or-kill check-in in #401, with #270 as the
        # automatic idle-timeout backstop.
        return (
            "##end_of_execution##" in line
            or "##execution_error##" in line
            or "(SyntaxError)" in line
        )