import re
import secrets
from pathlib import Path
from .subprocess_language import SubprocessLanguage

# irb prints this once per session when it starts echoing piped input. It is a
# session banner, not part of any block's echo, so it must not consume a line
# from the echo window.
_IRB_BANNER_RE = re.compile(r"Switch to(?: non)? inspect mode\.\s*$")
_ACTIVE_LINE_RE = re.compile(r"##active_line\d+##")

# The shared subprocess reader recognizes these strings, so Ruby's per-block
# markers are translated back to them before they leave the language layer.
_CANONICAL_END_MARKER = "##end_of_execution##"
_CANONICAL_ERROR_MARKER = "##execution_error##"


class Ruby(SubprocessLanguage):
    file_extension = "rb"
    name = "Ruby"

    def __init__(self):
        super().__init__()
        self.start_cmd = ["irb"]
        # The reader thread can see a line before preprocess_code() runs: the
        # live tests start the session first, and a restarted session prints
        # its banner immediately. Initialize the echo window and the markers
        # so those early lines are filtered instead of aborting the reader.
        self.code_line_count = 0
        self._end_marker = _CANONICAL_END_MARKER
        self._error_marker = _CANONICAL_ERROR_MARKER
        self._nonce = None
        self._active_marker_re = None

    def preprocess_code(self, code):
        """
        Add active line markers
        Wrap in a tryCatch for better error handling 
        Add end of execution marker
        """

        lines = code.split("\n")
        processed_lines = []

        # irb echoes the submitted block, comments included, before it runs
        # anything. A fixed marker can therefore be forged by a user comment
        # that matches it exactly: completion markers would end the block early
        # and an active-line marker would close the echo window, exposing the
        # next echoed comment (such as a completion marker) to the shared
        # reader. Make every marker unique to this block so an echoed comment
        # can never match one; line_postprocessor translates the block's own
        # markers back to the canonical strings the reader expects.
        nonce = secrets.token_hex(4)
        self._nonce = nonce
        self._end_marker = f"##end_of_execution_{nonce}##"
        self._error_marker = f"##execution_error_{nonce}##"
        self._active_marker_re = re.compile(rf"##active_line(\d+)_{nonce}##")

        for i, line in enumerate(lines, 1):
            # Add active line print
            processed_lines.append(f'puts "##active_line{i}_{nonce}##"')
            processed_lines.append(line)
        # Join lines to form the processed code
        processed_code = "\n".join(processed_lines)

        # Wrap in a tryCatch for error handling and add end of execution marker
        processed_code = f"""
begin
  {processed_code}
rescue => e
  puts "{self._error_marker}\\n" + e.message
ensure
  puts "{self._end_marker}\\n"
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
        # submitted block before it runs anything, comments included, so a
        # marker string taken from the block itself is ambiguous; the wrapper
        # emits per-block markers instead (see preprocess_code), and only those
        # bypass the window. That keeps a drifted window from swallowing the
        # completion signal without letting an echoed comment end the block.
        stripped = line.strip()
        if self.code_line_count > 0 and _IRB_BANNER_RE.fullmatch(stripped):
            # One-time session banner: consume it without spending a source line,
            # or the echoed block's last line would leak as output.
            return None
        # The per-block markers still bypass the echo window: if it drifts, the
        # completion line would otherwise be skipped and run() would wait
        # forever. Only the current block's marker matches, so an echoed user
        # comment spelling the historical marker cannot end execution early.
        if stripped == self._end_marker:
            self.code_line_count = 0
            return _CANONICAL_END_MARKER + "\n"
        if stripped.startswith(self._error_marker):
            self.code_line_count = 0
            return _CANONICAL_ERROR_MARKER + stripped[len(self._error_marker) :] + "\n"
        # Active-line markers carry the nonce too: an exact canonical marker is
        # an echoed user comment, so it must spend a window line instead of
        # closing the window and exposing the rest of the echo. Translate the
        # block's own marker back to the form detect_active_line() parses.
        if self._active_marker_re:
            match = self._active_marker_re.fullmatch(stripped)
            if match:
                self.code_line_count = 0
                return f"##active_line{match.group(1)}##\n"
        if _ACTIVE_LINE_RE.fullmatch(stripped):
            if self.code_line_count > 0:
                self.code_line_count -= 1
            return None
        # __init__ sets the counter, so it is always defined by the time the
        # reader can call this. Decrement and skip the echoed lines.
        if self.code_line_count > 0:
            self.code_line_count -= 1
            return None
        # irb prints a bare `nil` for a nil result. Match only that value so
        # strings such as "vanilla" or "nilpotent" survive as real output.
        if stripped in ("nil", "=> nil"):
            return None
        return line

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line or "##execution_error##" in line