import os
import queue
import re
import signal
import subprocess
import threading
import time
import traceback

from ..base_language import BaseLanguage


def _env_seconds(name, default):
    """Read a timeout (in seconds) from the environment. 0 / negative disables it."""
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else 0.0


# A command that produces no output for this long is treated as hung and killed.
# This is what stops `sshfs`, `ssh`, `apt` waiting on a prompt, etc. from blocking
# the agent forever. Commands that keep streaming output are never killed by this.
DEFAULT_IDLE_TIMEOUT = 120.0
# Absolute wall-clock cap. Disabled (0) by default so long *productive* builds run
# to completion; the idle timeout is what catches genuine hangs.
DEFAULT_TOTAL_TIMEOUT = 0.0


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

    def detect_active_line(self, line):
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

    def write_block_to_stdin(self, code):
        """Send a processed code block to the language subprocess."""
        payload = code if code.endswith("\n") else code + "\n"
        if self.binary_stdio:
            self.process.stdin.write(payload.encode("utf-8"))
        else:
            self.process.stdin.write(payload)
        self.process.stdin.flush()

    def _kill_process_group(self):
        """SIGKILL the shell *and every process it spawned*.

        ``Popen.terminate()`` only signals the shell itself, so a child that is
        blocking (sshfs, ssh, a package manager waiting on a prompt) survives and
        keeps the pipes open — which is precisely how a hung command wedged the
        whole interpreter.

        Safety: the process group is only signalled when it is genuinely NOT our
        own. If ``start_new_session`` ever failed, the child would share our group
        and ``killpg`` would kill Open Interpreter itself (and its parent shell).
        In that case fall back to killing just the child.
        """
        proc = self.process
        if not proc or proc.poll() is not None:
            return
        try:
            if os.name == "posix":
                pgid = os.getpgid(proc.pid)
                if pgid != os.getpgid(0):  # never signal our own group
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    def stop(self):
        """Halt a running command.

        ``BaseLanguage.stop()`` is a no-op, so before this override Ctrl-C (and
        ``Terminal.stop()``) could not interrupt a running shell command at all.
        """
        self._kill_process_group()
        self.done.set()

    def terminate(self):
        if self.process:
            self._kill_process_group()
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass
            try:
                self.process.wait(timeout=5)
            except Exception:
                pass
            self.process = None

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
        # Give the shell its own session/process group so the whole tree can be
        # killed together on timeout. Without this the child shares our group and
        # killpg() would take down Open Interpreter itself.
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
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

        idle_timeout = _env_seconds("INTERPRETER_COMMAND_IDLE_TIMEOUT", DEFAULT_IDLE_TIMEOUT)
        total_timeout = _env_seconds("INTERPRETER_COMMAND_TIMEOUT", DEFAULT_TOTAL_TIMEOUT)
        started_at = time.time()
        last_output_at = started_at

        def _timed_out():
            """Return a reason string if this command should be killed, else None."""
            now = time.time()
            if idle_timeout and (now - last_output_at) > idle_timeout:
                return (
                    f"produced no output for {idle_timeout:.0f}s",
                    idle_timeout,
                    "INTERPRETER_COMMAND_IDLE_TIMEOUT",
                )
            if total_timeout and (now - started_at) > total_timeout:
                return (
                    f"exceeded the {total_timeout:.0f}s total time limit",
                    total_timeout,
                    "INTERPRETER_COMMAND_TIMEOUT",
                )
            return None

        while True:
            if not self.output_queue.empty():
                yield self.output_queue.get()
                last_output_at = time.time()
            else:
                time.sleep(0.1)
            try:
                output = self.output_queue.get(timeout=0.3)  # Waits for 0.3 seconds
                yield output
                last_output_at = time.time()
            except queue.Empty:
                if self.done.is_set():
                    # Try to yank 3 more times from it... maybe there's something in there...
                    # (I don't know if this actually helps. Maybe we just need to yank 1 more time)
                    for _ in range(3):
                        if not self.output_queue.empty():
                            yield self.output_queue.get()
                        time.sleep(0.2)
                    break

                # Nothing arrived — check whether this command has hung. Without
                # this the loop waits forever for an end-of-execution marker that
                # a blocked command (sshfs, ssh, a prompt-waiting installer) will
                # never print.
                timed_out = _timed_out()
                if timed_out:
                    reason, limit, env_var = timed_out
                    self._kill_process_group()
                    self.done.set()
                    # Drain anything the command managed to emit before the kill.
                    while not self.output_queue.empty():
                        yield self.output_queue.get()
                    yield {
                        "type": "console",
                        "format": "output",
                        "content": (
                            f"\n[Open Interpreter] Command killed: it {reason}.\n"
                            f"The command and every process it started were terminated "
                            f"(SIGKILL to the process group).\n"
                            f"If this command legitimately needs longer, raise {env_var} "
                            f"(seconds, 0 disables), or run it in the background "
                            f"(e.g. append ' &' or use nohup) so it does not block.\n"
                        ),
                    }
                    # The shell is dead; the next run() will start a fresh one.
                    self.process = None
                    break

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
                    line = re.sub(r"##active_line\d+##", "", line)
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
                    if line:
                        self.output_queue.put(
                            {"type": "console", "format": "output", "content": line}
                        )
                elif self.detect_end_of_execution(line):
                    # Sometimes there's a little extra on the same line, so be sure to send that out
                    line = line.replace("##end_of_execution##", "").strip()
                    if line:
                        self.output_queue.put(
                            {"type": "console", "format": "output", "content": line}
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
        except ValueError as e:
            if "operation on closed file" in str(e):
                if self.verbose:
                    print("Stream closed while reading.")
            else:
                raise e
