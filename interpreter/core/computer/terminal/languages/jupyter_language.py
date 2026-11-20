"""
This is NOT jupyter language, this is just python. 
Gotta split this out, generalize it, and move all the python additions to python.py, which imports this
"""

import ast
import logging
import os
import queue
import re
import sys
import threading
import time
import traceback

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import litellm
from jupyter_client import KernelManager

from ..base_language import BaseLanguage

DEBUG_MODE = False

# A hard kernel exit produces no idle message, so the run has to say why it
# ended. The listener queues this before returning; the capture loop may also
# detect the exit itself, so both paths share one constant to keep it to a
# single notice per run.
_KERNEL_EXIT_NOTICE = "The Python kernel exited. It will restart on the next run."

# How long the capture loop waits for the listener to finish draining a dead
# kernel's remaining output. The listener's own drain is bounded at 0.5 s and
# polls the channel every 0.05 s; this covers that plus the capture poll
# interval. Waiting is bounded so a listener stuck in a read cannot hang the
# run, in which case capture reports the exit itself.
_KERNEL_EXIT_DRAIN_TIMEOUT = 0.8


def _is_kernel_exit_notice(chunk):
    """Whether an output chunk is the kernel-exit notice."""
    return (
        isinstance(chunk, dict)
        and chunk.get("type") == "console"
        and chunk.get("content") == _KERNEL_EXIT_NOTICE
    )


# When running from an executable, ipykernel calls itself infinitely
# This is a workaround to detect it and launch it manually
if "ipykernel_launcher" in sys.argv:
    if sys.path[0] == "":
        del sys.path[0]

    from ipykernel import kernelapp as app

    app.launch_new_instance()
    sys.exit(0)


def _display_data_chunk(data):
    """Map a display_data/execute_result data dict onto one output chunk.

    Every representation leaves a record: unknown types report themselves
    instead of vanishing, so dropped output never looks like no output.
    """
    if "image/png" in data:
        return {
            "type": "image",
            "format": "base64.png",
            "content": data["image/png"],
        }
    elif "image/jpeg" in data:
        return {
            "type": "image",
            "format": "base64.jpeg",
            "content": data["image/jpeg"],
        }
    elif "text/html" in data:
        return {
            "type": "code",
            "format": "html",
            "content": data["text/html"],
        }
    elif "text/plain" in data:
        return {
            "type": "console",
            "format": "output",
            "content": data["text/plain"],
        }
    elif "application/javascript" in data:
        return {
            "type": "code",
            "format": "javascript",
            "content": data["application/javascript"],
        }
    return {
        "type": "console",
        "format": "output",
        "content": (
            "(unrendered " + ", ".join(data) + " output)"
            if data
            else "(unrendered empty output)"
        ),
    }


class JupyterLanguage(BaseLanguage):
    file_extension = "py"
    name = "Python"
    aliases = ["py"]

    # msg_id of the execute_request currently being listened for, and a counter
    # that fences a lingering listener off once a newer request starts: a
    # listener only consumes the channel while its generation is current, so one
    # left over from an abandoned run exits instead of stealing a newer run's
    # messages. Class-level so a freshly built language is in a known state.
    _execution_id = None
    _listener_generation = 0
    # Set by the listener once it has drained a dead kernel's remaining output
    # and queued the exit notice after it, so capture can wait for that drain
    # instead of reporting the exit ahead of output still on its way.
    _kernel_exit_drain_done = None
    _kernel_exit_waited_for_drain = False

    def __init__(self, computer):
        self.computer = computer

        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self._wait_for_client()
        time.sleep(0.5)

        self.listener_thread = None
        self.finish_flag = False
        # True only while __init__ runs the matplotlib/display warm-up.
        # The listener must not start an input-patience model call then.
        self._startup_warmup = False
        # Whether the current request has been submitted and whether its
        # listener has started, for the narrow Ctrl-C window in between.
        self._request_submitted = False
        self._listener_started = False

        # DISABLED because sometimes this bypasses sending it up to us for some reason!
        # Give it our same matplotlib backend
        # backend = matplotlib.get_backend()

        # Use Agg, which bubbles everything up as an image.
        # Not perfect (I want interactive!) but it works.
        backend = "Agg"

        code = f"""
import matplotlib
matplotlib.use('{backend}')
        """.strip()

        # Use Inline actually, it's better I think
        code = """
%matplotlib inline
import matplotlib.pyplot as plt
""".strip()

        if not self._kernel_is_alive():
            # A replacement that died during startup cannot take the setup
            # code. Skipping the warm-up leaves run() to report the failure
            # instead of restarting recursively through __init__.
            return

        self._startup_warmup = True
        try:
            try:
                for _ in self.run(code):
                    pass
            except KeyboardInterrupt:
                # __init__ has not returned, so Terminal cannot clean up this
                # half-constructed instance. Terminate the kernel before the
                # interrupt aborts construction and leaks it.
                self._cleanup_after_interrupted_startup()
                raise
        finally:
            self._startup_warmup = False

        # DISABLED because it doesn't work??
        # Disable color outputs in the terminal, which don't look good in OI and aren't useful
        # code = """
        # from IPython.core.getipython import get_ipython
        # get_ipython().colors = 'NoColor'
        # """
        # self.run(code)

    def terminate(self):
        self.kc.stop_channels()
        self.km.shutdown_kernel()

    def _cleanup_after_interrupted_startup(self):
        """Stop the listener and kernel when Ctrl-C interrupts __init__ warm-up."""
        self.finish_flag = True
        listener = getattr(self, "listener_thread", None)
        if listener is not None and listener.is_alive():
            listener.join(timeout=10)
        self.terminate()

    def _kernel_is_alive(self):
        """Whether the kernel process is still running.

        Client channels can stay open after the process exits, so checking
        `kc.is_alive()` is not enough: `os._exit(0)` leaves them open and the
        run waits forever for an idle message that cannot arrive.
        """
        km = getattr(self, "km", None)
        if km is None:
            return True
        try:
            return bool(km.is_alive())
        except Exception:
            return True

    def _stop_kernel(self):
        """Stop channels and shut the kernel down, ignoring teardown failures."""
        try:
            if getattr(self, "kc", None):
                self.kc.stop_channels()
        except Exception:
            pass
        try:
            if getattr(self, "km", None):
                self.km.shutdown_kernel(now=True)
        except Exception:
            pass

    def _fence_listener(self, timeout=5):
        """Invalidate and join the previous listener before the client changes.

        A listener blocked in the input-patience model call only notices a stop
        signal once that call returns, and it holds its own reference to the
        client it was built for. Fencing it by generation is what guarantees it
        cannot send input to, consume output from, or stop a later run even if
        the bounded join below times out.
        """
        self._listener_generation = getattr(self, "_listener_generation", 0) + 1
        previous_listener = getattr(self, "listener_thread", None)
        if previous_listener is not None and previous_listener.is_alive():
            previous_listener.join(timeout=timeout)

    def _ensure_kernel(self):
        """Recreate the kernel after a hard exit, with OI's setup applied.

        Re-running __init__ is deliberate: a bare restart would leave the new
        kernel without the matplotlib backend and the display formatter, and
        the model depends on both.
        """
        if self._kernel_is_alive():
            return
        self.finish_flag = True
        self._fence_listener()
        self._stop_kernel()
        self.__init__(self.computer)

    def _wait_for_client(self, timeout=10):
        """Wait for the client channels to come up, but never forever.

        A replacement kernel that dies during startup never brings its client
        up, so its process is checked too: a dead kernel ends the wait instead
        of idling until the deadline.
        """
        deadline = time.time() + timeout
        while not self.kc.is_alive():
            if not self._kernel_is_alive() or time.time() >= deadline:
                return
            time.sleep(0.1)

    def run(self, code):
        try:
            self._ensure_kernel()
        except Exception:
            # A restart that fails midway (start_kernel, channels, ...) must be
            # reported like any other run failure, not escape as a traceback;
            # tear down whatever part of the replacement came up.
            self._stop_kernel()
            yield {"type": "console", "format": "output", "content": traceback.format_exc()}
            return
        if not self._kernel_is_alive():
            yield {
                "type": "console",
                "format": "output",
                "content": "The Python kernel could not be restarted.",
            }
            return
        self._wait_for_client()

        self.last_output_time = time.time()
        self.last_output_message_time = time.time()

        ################################################################
        ### OFFICIAL OPEN INTERPRETER GOVERNMENT ISSUE SKILL LIBRARY ###
        ################################################################

        # try:
        #     functions = string_to_python(code)
        # except:
        #     # Non blocking
        #     functions = {}

        # if self.computer.save_skills and functions:
        #     skill_library_path = self.computer.skills.path

        #     if not os.path.exists(skill_library_path):
        #         os.makedirs(skill_library_path)

        #     for filename, function_code in functions.items():
        #         with open(f"{skill_library_path}/{filename}.py", "w") as file:
        #             file.write(function_code)

        self.finish_flag = False
        try:
            try:
                preprocessed_code = self.preprocess_code(code)
            except Exception:
                # Any errors produced here are our fault.
                # Also, for python, you don't need them! It's just for active_line and stuff. Just looks pretty.
                # Exception-only: a bare except also catches KeyboardInterrupt,
                # which would fall through and execute the block the user just
                # cancelled. The interrupt has to reach the handler below.
                preprocessed_code = code
            message_queue = queue.Queue()
            self._execute_code(preprocessed_code, message_queue)
            yield from self._capture_output(message_queue)
        except GeneratorExit:
            raise  # gotta pass this up!
        except KeyboardInterrupt:
            # Ctrl-C is the user stopping this block, not the block failing. The
            # kernel has its own session, so the terminal's SIGINT never reaches
            # it — flag the listener to interrupt it, the way stop() does, then
            # let the interrupt travel up to the terminal's own handler.
            self.stop()
            if getattr(self, "_request_submitted", False) and not getattr(
                self, "_listener_started", False
            ):
                # Ctrl-C landed after execute() submitted the block but before
                # its listener started. stop() only sets finish_flag, so no
                # listener will send the interrupt; do it here.
                self.km.interrupt_kernel()
            raise
        except Exception:
            content = traceback.format_exc()
            yield {"type": "console", "format": "output", "content": content}

    def _queue_iopub_output(self, msg, message_queue):
        """Route one IOPub message's output to the message queue."""
        content = msg["content"]

        if msg["msg_type"] == "stream":
            line, active_line = self.detect_active_line(content["text"])
            if active_line:
                message_queue.put(
                    {
                        "type": "console",
                        "format": "active_line",
                        "content": active_line,
                    }
                )
            message_queue.put(
                {"type": "console", "format": "output", "content": line}
            )
        elif msg["msg_type"] == "error":
            content = "\n".join(content["traceback"])
            # Remove color codes
            ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
            content = ansi_escape.sub("", content)
            message_queue.put(
                {
                    "type": "console",
                    "format": "output",
                    "content": content,
                }
            )
        elif msg["msg_type"] in ["display_data", "execute_result"]:
            message_queue.put(_display_data_chunk(content["data"]))

    def _report_kernel_exit(self, kc, execution_id, message_queue, drain_done=None):
        """Queue the exit notice after draining output the kernel already sent.

        The process can die between the last read and this check, so messages
        already delivered to the channel still belong to the run that just
        ended. They are drained here, on the thread that owns the socket, so
        the output that led to the exit is reported before the notice instead
        of being dropped behind it. The drain is bounded because the channel
        may already be closed. An empty read does not end the drain: the kernel
        can die after sending output, so the loop keeps polling until the
        deadline and only a channel that raises ends it early.

        ``drain_done`` is set after the notice is queued so the capture loop
        can wait for the drain rather than report the exit first and leave
        output queued behind a run that already ended.
        """
        deadline = time.time() + 0.5
        while time.time() < deadline:
            try:
                msg = kc.iopub_channel.get_msg(timeout=0.05)
            except queue.Empty:
                # The kernel died after sending output, so a message can
                # arrive after an empty read. Poll until the deadline instead
                # of treating the first empty read as the end of the drain.
                continue
            except Exception:
                # A closed channel raises instead of returning Empty; keep the
                # early exit so a dead socket does not stall the notice.
                break

            parent_id = (msg.get("parent_header") or {}).get("msg_id")
            if parent_id and parent_id != execution_id:
                continue

            self.last_output_time = time.time()
            if (
                msg["header"]["msg_type"] == "status"
                and msg["content"]["execution_state"] == "idle"
            ):
                break
            self._queue_iopub_output(msg, message_queue)

        message_queue.put(
            {
                "type": "console",
                "format": "output",
                "content": _KERNEL_EXIT_NOTICE,
            }
        )
        self.finish_flag = True
        if drain_done is not None:
            # After the notice, so a capture waiting on this event finds the
            # drained output and the notice already queued.
            drain_done.set()

    def _execute_code(self, code, message_queue):
        # An earlier run's listener may still be alive if its generator was
        # abandoned before it exited. It reads the same iopub channel, so
        # stop and join it first — otherwise it would steal this run's
        # messages and could set finish_flag on this run's idle status.
        previous_listener = getattr(self, "listener_thread", None)
        if previous_listener is not None and previous_listener.is_alive():
            # Only clear the flag if this stop signal is ours: a caller may
            # have set finish_flag to request an interrupt, and that request
            # must still be honored by the new listener.
            already_stopping = getattr(self, "finish_flag", False)
            self.finish_flag = True
            previous_listener.join(timeout=5)
            if previous_listener.is_alive():
                # Still stuck (e.g. inside the input-patience model call,
                # which only notices the flag once it returns). Keep the stop
                # signal set and give it more time — sending the next request
                # now would let it consume this run's messages from the
                # shared channel. Bounded so one stuck thread cannot hang
                # new runs forever; the generation check below keeps any
                # overlap from corrupting the new run.
                for _ in range(5):
                    previous_listener.join(timeout=5)
                    if not previous_listener.is_alive():
                        break
            if not already_stopping:
                self.finish_flag = False

        # Every request gets a generation number. A listener only consumes
        # the channel while its generation is current; once a newer request
        # starts, a lingering listener exits instead of stealing messages.
        generation = getattr(self, "_listener_generation", 0) + 1
        self._listener_generation = generation
        self._request_submitted = False
        self._listener_started = False
        # Per-request handshake with the listener's dead-kernel drain. A fresh
        # event fences a lingering listener from an earlier run, which holds
        # its own bound event rather than reading this attribute.
        drain_done = threading.Event()
        self._kernel_exit_drain_done = drain_done
        self._kernel_exit_waited_for_drain = False

        # Send the request first so its message id is known before anything
        # listens. Every iopub message carries the id of the request that
        # caused it in parent_header, which is the only way to tell this
        # execution's output from an earlier one's.
        execution_id = self.kc.execute(code)
        self._request_submitted = True
        self._execution_id = execution_id

        def iopub_message_listener(
            execution_id=execution_id,
            generation=generation,
            kc=self.kc,
            drain_done=drain_done,
        ):
            # Bound as defaults so this listener always filters against the
            # request id from its own execute call, even if a later run
            # overwrites self._execution_id. The client is bound for the same
            # reason: a restart swaps self.kc, and an old listener must never
            # read from or type into the replacement.
            max_retries = 100
            while True:
                if generation != self._listener_generation:
                    # A newer request started — or a restart replaced the
                    # kernel — so stop consuming the channel and leave every
                    # shared flag and the client to its current owner. This
                    # check comes before the liveness one: during a restart
                    # the process is briefly dead, and a stale listener that
                    # reported that would stop the recovered run.
                    return
                # A hard exit (os._exit, OOM, segfault) closes the channels
                # without sending idle; stop waiting and report it.
                if not self._kernel_is_alive():
                    self._report_kernel_exit(
                        kc, execution_id, message_queue, drain_done
                    )
                    return
                # If self.finish_flag = True, and we didn't set it (we do below), we need to stop. That's our "stop"
                if self.finish_flag == True:
                    if DEBUG_MODE:
                        print("interrupting kernel!!!!!")
                    self.km.interrupt_kernel()
                    return
                # For async usage
                if (
                    hasattr(self.computer.interpreter, "stop_event")
                    and self.computer.interpreter.stop_event.is_set()
                ):
                    self.km.interrupt_kernel()
                    self.finish_flag = True
                    return
                try:
                    input_patience = int(
                        os.environ.get("INTERPRETER_TERMINAL_INPUT_PATIENCE", 15)
                    )
                    if (
                        not getattr(self, "_startup_warmup", False)
                        and time.time() - self.last_output_time > input_patience
                        and time.time() - self.last_output_message_time > input_patience
                    ):
                        self.last_output_message_time = time.time()

                        text = f"{self.computer.interpreter.messages}\n\nThe program above has been running for over 15 seconds. It might require user input. Are there keystrokes that the user should type in, to proceed after the last command?"
                        if time.time() - self.last_output_time > 500:
                            text += f" If you think the process is frozen, or that the user wasn't expect it to run for this long (it has been {time.time() - self.last_output_time} seconds since last output) then say <input>CTRL-C</input>."

                        messages = [
                            {
                                "role": "system",
                                "type": "message",
                                "content": "You are an expert programming assistant. You will help the user determine if they should enter input into the terminal, per the user's requests. If you think the user would want you to type something into stdin, enclose it in <input></input> XML tags, like <input>y</input> to type 'y'.",
                            },
                            {"role": "user", "type": "message", "content": text},
                        ]
                        params = {
                            "messages": messages,
                            "model": self.computer.interpreter.llm.model,
                            "stream": True,
                            "temperature": 0,
                        }
                        if self.computer.interpreter.llm.api_key:
                            params["api_key"] = self.computer.interpreter.llm.api_key

                        response = ""
                        for chunk in litellm.completion(**params):
                            content = chunk.choices[0].delta.content
                            if type(content) == str:
                                response += content

                        if generation != self._listener_generation:
                            # A newer request started — or a restart replaced
                            # the kernel — while the model call was blocking.
                            # The answer describes an execution that no longer
                            # owns the client, so act on none of it.
                            return

                        # Parse the response for input tags
                        input_match = re.search(r"<input>(.*?)</input>", response)
                        if input_match:
                            user_input = input_match.group(1)
                            # Check if the user input is CTRL-C
                            self.finish_flag = True
                            if user_input.upper() == "CTRL-C":
                                self.finish_flag = True
                            else:
                                kc.input(user_input)

                    msg = kc.iopub_channel.get_msg(timeout=0.05)
                except queue.Empty:
                    continue
                except Exception as e:
                    max_retries -= 1
                    if max_retries < 0:
                        raise
                    print("Jupyter error, retrying:", str(e))
                    continue

                # An earlier execution's messages can still be in the channel:
                # interrupt_kernel() returns before the kernel has finished
                # emitting, and stop() abandons the listener without draining.
                # parent_header names the request that produced this message,
                # so anything that is not this execution's is skipped — before
                # last_output_time, so stale traffic cannot pass for activity.
                parent_id = (msg.get("parent_header") or {}).get("msg_id")
                if parent_id and parent_id != execution_id:
                    continue

                self.last_output_time = time.time()

                if DEBUG_MODE:
                    print("-----------" * 10)
                    print("Message received:", msg["content"])
                    print("-----------" * 10)

                if (
                    msg["header"]["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    # Set finish_flag and return when the kernel becomes idle
                    if DEBUG_MODE:
                        print("from thread: kernel is idle")
                    self.finish_flag = True
                    return

                self._queue_iopub_output(msg, message_queue)

        self.listener_thread = threading.Thread(target=iopub_message_listener)
        # self.listener_thread.daemon = True
        self.listener_thread.start()
        self._listener_started = True

        if DEBUG_MODE:
            print(
                "thread is on:", self.listener_thread.is_alive(), self.listener_thread
            )

    def detect_active_line(self, line):
        if "##active_line" in line:
            # Split the line by "##active_line" and grab the last element
            last_active_line = line.split("##active_line")[-1]
            # Split the last active line by "##" and grab the first element
            try:
                active_line = int(last_active_line.split("##")[0])
            except Exception:
                # Not a bare except: Ctrl-C during this parse must propagate.
                active_line = 0
            # Remove all ##active_line{number}##\n
            line = re.sub(r"##active_line\d+##\n", "", line)
            return line, active_line
        return line, None

    def _capture_output(self, message_queue):
        exit_notice_delivered = False

        while True:
            time.sleep(0.1)

            # For async usage
            if (
                hasattr(self.computer.interpreter, "stop_event")
                and self.computer.interpreter.stop_event.is_set()
            ):
                self.finish_flag = True
                break

            if self.listener_thread:
                try:
                    output = message_queue.get(timeout=0.1)
                    if DEBUG_MODE:
                        print(output)
                    if _is_kernel_exit_notice(output):
                        exit_notice_delivered = True
                    yield output

                except queue.Empty:
                    if self.finish_flag:
                        time.sleep(0.1)

                        try:
                            output = message_queue.get(timeout=0.1)
                            if DEBUG_MODE:
                                print(output)
                            if _is_kernel_exit_notice(output):
                                exit_notice_delivered = True
                            yield output
                        except queue.Empty:
                            if DEBUG_MODE:
                                print("we're done")
                            break
                    elif not self._kernel_is_alive():
                        # The listener drains the output the kernel sent
                        # before it died and only queues its notice after
                        # that output. Capture can see the empty queue and
                        # the dead kernel first; without waiting it would
                        # report the exit and end the run while the listener
                        # still holds drained output that then has no reader.
                        # The wait is bounded so a listener stuck in a read
                        # cannot hang the run; on timeout capture falls back
                        # to draining and reporting the exit itself.
                        if (
                            self._kernel_exit_drain_done is not None
                            and not self._kernel_exit_waited_for_drain
                        ):
                            self._kernel_exit_waited_for_drain = True
                            self._kernel_exit_drain_done.wait(
                                _KERNEL_EXIT_DRAIN_TIMEOUT
                            )
                            continue
                        # The listener may have died mid-drain; drain what it
                        # left and deliver the notice once if it was not
                        # already in the queue.
                        while not message_queue.empty():
                            output = message_queue.get()
                            if _is_kernel_exit_notice(output):
                                exit_notice_delivered = True
                            yield output
                        if not exit_notice_delivered:
                            yield {
                                "type": "console",
                                "format": "output",
                                "content": _KERNEL_EXIT_NOTICE,
                            }
                        self.finish_flag = True
                        break
                    elif (
                        not self.listener_thread.is_alive()
                        and message_queue.empty()
                    ):
                        if self.finish_flag:
                            # The listener recorded normal completion between
                            # the check above and this branch. The queue is
                            # empty because the run finished, not because the
                            # kernel is stuck; interrupting here would kill a
                            # healthy kernel and report a failed execution.
                            break
                        # The listener died without setting finish_flag. The
                        # kernel may still be running code whose output this
                        # run can no longer collect, so interrupt it and
                        # report why the run ended instead of pretending it
                        # completed normally.
                        try:
                            self.km.interrupt_kernel()
                        except Exception:
                            # Best-effort interrupt: a dead client connection
                            # must not turn the report into another crash.
                            pass
                        self.finish_flag = True
                        yield {
                            "type": "console",
                            "format": "output",
                            "content": (
                                "The Python kernel stopped sending output; "
                                "the run was interrupted."
                            ),
                        }
                        break

    def stop(self):
        self.finish_flag = True

    def preprocess_code(self, code):
        return preprocess_python(code)


def preprocess_python(code):
    """
    Add active line markers
    Wrap in a try except
    """

    code = code.strip()

    # Add print commands that tell us what the active line is
    # but don't do this if any line starts with ! or %
    if (
        not any(line.strip().startswith(("!", "%")) for line in code.split("\n"))
        and os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower()
        == "true"
    ):
        code = add_active_line_prints(code)

    # Wrap in a try except (DISABLED)
    # code = wrap_in_try_except(code)

    # Remove any whitespace lines, as this will break indented blocks
    # (are we sure about this? test this)
    code_lines = code.split("\n")
    code_lines = [c for c in code_lines if c.strip() != ""]
    code = "\n".join(code_lines)

    return code


def add_active_line_prints(code):
    """
    Add print statements indicating line numbers to a python string.
    """
    # Replace newlines and comments with pass statements, so the line numbers are accurate (ast will remove them otherwise)
    code_lines = code.split("\n")
    in_multiline_string = False
    for i in range(len(code_lines)):
        line = code_lines[i]
        if '"""' in line or "'''" in line:
            in_multiline_string = not in_multiline_string
        if not in_multiline_string and (line.strip().startswith("#") or line == ""):
            whitespace = len(line) - len(line.lstrip(" "))
            code_lines[i] = " " * whitespace + "pass"
    processed_code = "\n".join(code_lines)
    try:
        tree = ast.parse(processed_code)
    except Exception:
        # If you can't parse the processed version, try the unprocessed version before giving up.
        # Exception-only so a KeyboardInterrupt raised while parsing is not
        # swallowed and turned into a normal return.
        tree = ast.parse(code)
    transformer = AddLinePrints()
    new_tree = transformer.visit(tree)
    return ast.unparse(new_tree)


class AddLinePrints(ast.NodeTransformer):
    """
    Transformer to insert print statements indicating the line number
    before every executable line in the AST.
    """

    def insert_print_statement(self, line_number):
        """Inserts a print statement for a given line number."""
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[ast.Constant(value=f"##active_line{line_number}##")],
                keywords=[],
            )
        )

    def process_body(self, body):
        """Processes a block of statements, adding print calls."""
        new_body = []

        # In case it's not iterable:
        if not isinstance(body, list):
            body = [body]

        for sub_node in body:
            if hasattr(sub_node, "lineno"):
                new_body.append(self.insert_print_statement(sub_node.lineno))
            new_body.append(sub_node)

        return new_body

    def visit(self, node):
        """Overridden visit to transform nodes."""
        new_node = super().visit(node)

        # If node has a body, process it
        if hasattr(new_node, "body"):
            new_node.body = self.process_body(new_node.body)

        # If node has an orelse block (like in for, while, if), process it
        if hasattr(new_node, "orelse") and new_node.orelse:
            new_node.orelse = self.process_body(new_node.orelse)

        # Special case for Try nodes as they have multiple blocks
        if isinstance(new_node, ast.Try):
            for handler in new_node.handlers:
                handler.body = self.process_body(handler.body)
            if new_node.finalbody:
                new_node.finalbody = self.process_body(new_node.finalbody)

        return new_node


def wrap_in_try_except(code):
    # Add import traceback
    code = "import traceback\n" + code

    # Parse the input code into an AST
    parsed_code = ast.parse(code)

    # Wrap the entire code's AST in a single try-except block
    try_except = ast.Try(
        body=parsed_code.body,
        handlers=[
            ast.ExceptHandler(
                type=ast.Name(id="Exception", ctx=ast.Load()),
                name=None,
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="traceback", ctx=ast.Load()),
                                attr="print_exc",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        )
                    ),
                ],
            )
        ],
        orelse=[],
        finalbody=[],
    )

    # Assign the try-except block as the new body
    parsed_code.body = [try_except]

    # Convert the modified AST back to source code
    return ast.unparse(parsed_code)


def string_to_python(code_as_string):
    parsed_code = ast.parse(code_as_string)

    # Initialize containers for different categories
    import_statements = []
    functions = []
    functions_dict = {}

    # Traverse the AST
    for node in ast.walk(parsed_code):
        # Check for import statements
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for alias in node.names:
                # Handling the alias in import statements
                if alias.asname:
                    import_statements.append(f"import {alias.name} as {alias.asname}")
                else:
                    import_statements.append(f"import {alias.name}")
        # Check for function definitions
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                # ignore private functions
                continue

            # Keep the whole definition (signature, defaults, annotations,
            # decorators and body) rather than rebuilding a zero-argument def.
            functions.append({"name": node.name, "code": ast.unparse(node)})

    for func in functions:
        # Consolidating import statements and function definition
        function_content = "\n".join(import_statements) + "\n\n"
        function_content += func["code"] + "\n"

        # Adding to dictionary
        functions_dict[func["name"]] = function_content

    return functions_dict
