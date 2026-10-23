import os
import queue
import re
import subprocess
import threading
import time
import traceback

from ..base_language import BaseLanguage

_ACTIVE_LINE_MARKER_RE = re.compile(r"##active_line(\d+)##")
# Block-terminating markers. Both are stripped from the line that carries them so
# neither reaches the user as literal text.
_END_OF_EXECUTION_RE = re.compile(r"##end_of_execution##")
_EXECUTION_ERROR_RE = re.compile(r"##execution_error##")


class SubprocessLanguage(BaseLanguage):
    # Perl REPL uses a custom __OI_END__ block marker; text=True on Windows turns
    # \n into \r\n and the REPL waits forever. Subclasses set True for byte pipes.
    binary_stdio = False

    def __init__(self):
        self.start_cmd = []
        self.process = None
        self.verbose = False
        self.output_queue = queue.Queue()
        self.done = threading.Event()
        # Set by _abort_stream when the reader fails unexpectedly, so the
        # failure can be inspected after the turn rather than only being logged.
        self._stream_error = None

    def detect_active_line(self, line):
        """
        Return the line number from a complete numeric ##active_lineN## marker.

        Command output can legitimately contain placeholder text such as
        ##active_lineN##, so only complete numeric markers are control markers.
        """
        match = _ACTIVE_LINE_MARKER_RE.search(line)
        if match is None:
            return None
        return int(match.group(1))

    def detect_exit_code(self, line):
        """Exit status encoded in the end-of-execution marker, or None.

        Only subclasses whose marker carries the status (currently Bash)
        override this; the default keeps every other language unchanged.
        """
        return None

    def detect_end_of_execution(self, line):
        return None

    def line_postprocessor(self, line):
        return line

    def preprocess_code(self, code):
        """
        This needs to insert an end_of_execution marker of some kind,
        which can be detected by detect_end_of_execution.

        Optionally, add active line markers for detect_active_line.
        """
        return code

    # POSIX shells can run a command with its own stdin pointed at the null
    # device -- `{ cat; } < /dev/null` -- so the command sees EOF immediately
    # instead of blocking forever on a read that nothing will satisfy. Kept off
    # by default because it needs shell syntax: cmd.exe cannot parse braces, and
    # each shell needs its own form.
    #
    # Why not redirect the subprocess's stdin at Popen instead: that pipe is how
    # commands reach a persistent shell, so closing it ends the session. The
    # redirect is scoped to the command body, leaving the shell's own stdin --
    # the command channel -- intact. Verified: the next command still runs.
    #
    # Cost is that a bare `read` now returns 1 at EOF, so `read x && cmd` skips
    # cmd. That is visible in the output, which is the failure mode we want;
    # hanging is not.
    command_stdin_from_null_device = False
    null_device_path = "/dev/null"

    def _wrap_command_for_null_stdin(self, code):
        # The terminator differs by whether `code` already ends in a newline. A
        # newline followed by `;` is a bash syntax error ("near unexpected token
        # `;'"), and multi-line blocks do end in one. Single-line blocks need the
        # `;` added explicitly, since `{ cmd }` alone is a syntax error.
        terminator = "} " if code.endswith("\n") else "; } "
        return "{ " + code + terminator + "< " + self.null_device_path

    def write_block_to_stdin(self, code):
        """Send a processed code block to the language subprocess."""
        if self.command_stdin_from_null_device:
            code = self._wrap_command_for_null_stdin(code)
        payload = code if code.endswith("\n") else code + "\n"
        if self.binary_stdio:
            self.process.stdin.write(payload.encode("utf-8"))
        else:
            self.process.stdin.write(payload)
        self.process.stdin.flush()

    def terminate(self):
        if self.process:
            self.process.terminate()
            self.process.stdin.close()
            self.process.stdout.close()

    def start_process(self):
        if self.process:
            self.terminate()

        my_env = os.environ.copy()
        my_env["PYTHONIOENCODING"] = "utf-8"
        popen_kwargs = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "bufsize": 0,
            "env": my_env,
        }
        if self.binary_stdio:
            self.process = subprocess.Popen(self.start_cmd, **popen_kwargs)
        else:
            self.process = subprocess.Popen(
                self.start_cmd,
                text=True,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
                **popen_kwargs,
            )
        threading.Thread(
            target=self.handle_stream_output,
            args=(self.process.stdout, False),
            daemon=True,
        ).start()
        threading.Thread(
            target=self.handle_stream_output,
            args=(self.process.stderr, True),
            daemon=True,
        ).start()

    def run(self, code):
        retry_count = 0
        max_retries = 3

        # Setup
        try:
            code = self.preprocess_code(code)
            if not self.process:
                self.start_process()
        except:
            yield {
                "type": "console",
                "format": "output",
                "content": traceback.format_exc(),
            }
            return

        while retry_count <= max_retries:
            if self.verbose:
                print(f"(after processing) Running processed code:\n{code}\n---")

            self.done.clear()

            try:
                self.write_block_to_stdin(code)
                break
            except:
                if retry_count != 0:
                    # For UX, I like to hide this if it happens once. Obviously feels better to not see errors
                    # Most of the time it doesn't matter, but we should figure out why it happens frequently with:
                    # applescript
                    yield {
                        "type": "console",
                        "format": "output",
                        "content": f"{traceback.format_exc()}\nRetrying... ({retry_count}/{max_retries})\nRestarting process.",
                    }

                self.start_process()

                retry_count += 1
                if retry_count > max_retries:
                    yield {
                        "type": "console",
                        "format": "output",
                        "content": "Maximum retries reached. Could not execute code.",
                    }
                    return

        while True:
            if not self.output_queue.empty():
                yield self.output_queue.get()
            else:
                time.sleep(0.1)
            try:
                output = self.output_queue.get(timeout=0.3)  # Waits for 0.3 seconds
                yield output
            except queue.Empty:
                if self.done.is_set():
                    # Try to yank 3 more times from it... maybe there's something in there...
                    # (I don't know if this actually helps. Maybe we just need to yank 1 more time)
                    for _ in range(3):
                        if not self.output_queue.empty():
                            yield self.output_queue.get()
                        time.sleep(0.2)
                    break

        # Persistent shells report where they are after every command, the way
        # JupyterLanguage reports CWD/variables for Python. Opt-in via the
        # method existing, so languages without tracked state are unaffected.
        state_line = getattr(self, "_state_line", None)
        if callable(state_line):
            line = state_line()
            if line:
                yield {"type": "console", "format": "output", "content": line}

    def handle_stream_output(self, stream, is_error_stream):
        try:
            eof = b"" if self.binary_stdio else ""
            for raw_line in iter(stream.readline, eof):
                if self.verbose:
                    print(f"Received output line:\n{raw_line}\n---")

                if self.binary_stdio:
                    line = raw_line.decode("utf-8", errors="replace")
                else:
                    line = raw_line

                line = self.line_postprocessor(line)

                if line is None:
                    continue  # `line = None` is the postprocessor's signal to discard completely

                if self.detect_active_line(line):
                    active_line = self.detect_active_line(line)
                    # Sometimes there's a little extra on the same line, so be sure to send that out
                    line = _ACTIVE_LINE_MARKER_RE.sub("", line)
                    active_line_enabled = (
                        os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower()
                        == "true"
                    )
                    if active_line_enabled:
                        self.output_queue.put(
                            {
                                "type": "console",
                                "format": "active_line",
                                "content": active_line,
                            }
                        )
                    # The marker is usually printed on a line of its own, so what
                    # is left after removing it is just the newline that
                    # terminated it. A whitespace-only remainder carries no
                    # output and would show up as a stray blank line.
                    if line.strip():
                        self.output_queue.put(
                            {"type": "console", "format": "output", "content": line}
                        )
                elif self.detect_end_of_execution(line):
                    exit_code = self.detect_exit_code(line)
                    # Both markers end the block, so both must be stripped. An
                    # error marker that reaches the user verbatim reads as
                    # protocol noise rather than the error itself.
                    # \d* also consumes the exit status when the marker carries
                    # one (see Bash.detect_exit_code).
                    line = re.sub(r"##end_of_execution##\d*", "", line)
                    line = _EXECUTION_ERROR_RE.sub("", line).strip()
                    if line:
                        self.output_queue.put(
                            {"type": "console", "format": "output", "content": line}
                        )
                    # Only report failures: a successful command stays silent and
                    # costs the model nothing.
                    if exit_code:
                        self.output_queue.put(
                            {
                                "type": "console",
                                "format": "output",
                                "content": f"[exited with code {exit_code}]",
                            }
                        )
                    self.done.set()
                elif is_error_stream and "KeyboardInterrupt" in line:
                    self.output_queue.put(
                        {
                            "type": "console",
                            "format": "output",
                            "content": "KeyboardInterrupt",
                        }
                    )
                    time.sleep(0.1)
                    self.done.set()
                else:
                    self.output_queue.put(
                        {"type": "console", "format": "output", "content": line}
                    )
            # The stream hit EOF without the end marker: the shell itself exited
            # (user code ran `exit`, or `set -e`/errexit killed it before the
            # appended capture commands ran), or it closed its stdout and lives
            # on (handled below). A per-block EXIT trap cannot cover
            # this -- the shell is persistent across blocks, so a trap would only
            # fire when the whole session ends, not per block. Without this epilogue
            # `done` is never set and run() spins forever. Report the shell's
            # status and end the turn; the next block restarts the dead process via
            # the write retry in run(). Only the stdout reader reports: both
            # streams hit EOF on process death, and one report is enough.
            process = self.process
            if (
                not is_error_stream
                and not self.done.is_set()
                # Only the persistent shell's own stdout speaks for the shell.
                # A reader that outlived its process would end its successor's
                # block, and a language that runs a fresh process per block
                # (Java) reaches EOF without the marker every time and ends the
                # block itself.
                and process is not None
                and process.stdout is stream
            ):
                returncode = None
                # A shell closes its pipes while exiting, before it can be
                # reaped, so a single poll() here usually still sees it
                # running. Give a dying shell a moment to finish.
                try:
                    returncode = process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                if returncode is not None:
                    detail = f"the shell exited with code {returncode}"
                else:
                    # Alive with stdout at EOF: user code moved the shell's own
                    # stdout (`exec >file`). Every later end marker goes there
                    # too and no reader is left on the pipe, so the next block
                    # would hang. The pipe cannot be reattached from outside;
                    # replace the shell instead, via run()'s start-if-missing.
                    self.terminate()
                    self.process = None
                    detail = "the shell closed its output (e.g. `exec >file`) and was stopped"
                self.output_queue.put(
                    {
                        "type": "console",
                        "format": "output",
                        "content": f"\n[{detail}; it will restart on the next command.]",
                    }
                )
                self.done.set()
        except ValueError as e:
            if "operation on closed file" in str(e):
                if self.verbose:
                    print("Stream closed while reading.")
            else:
                self._abort_stream(e)
        except Exception as e:
            self._abort_stream(e)

    def _abort_stream(self, error):
        """End the turn after an unexpected reader failure, and report it.

        Without this the reader thread just dies. `done` is then never set --
        it is only set by the end-of-execution marker or the KeyboardInterrupt
        branch, both of which live in this method -- so run()'s consumer loop
        spins on `while True` forever, the turn never ends, and whatever the
        command had already printed stays unread in the pipe.

        The visible symptom is output that vanished, plus a terminal that stops
        responding to Ctrl-C, because the Rich Live display stays up for the
        whole hang. That was reported as "the previous attempt's output got
        swallowed" and mistaken for the model losing interest.

        The failure is surfaced rather than swallowed: a reader that dies quietly
        looks identical to a command that produced no output. Whatever was
        already queued has been yielded by this point, so only the abort itself
        needs reporting.
        """
        self._stream_error = error
        self.output_queue.put(
            {
                "type": "console",
                "format": "output",
                "content": (
                    f"\n[output stream stopped unexpectedly: "
                    f"{type(error).__name__}: {error}]"
                ),
            }
        )
        # Ends the consumer loop. Set last, so the diagnostic above is already
        # queued and gets drained rather than lost to the teardown.
        self.done.set()
