"""
This is NOT jupyter language, this is just python.
Gotta split this out, generalize it, and move all the python additions to python.py, which imports this
"""

import ast
import hashlib
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

# When running from an executable, ipykernel calls itself infinitely
# This is a workaround to detect it and launch it manually
if "ipykernel_launcher" in sys.argv:
    if sys.path[0] == "":
        del sys.path[0]

    from ipykernel import kernelapp as app

    app.launch_new_instance()
    sys.exit(0)


class JupyterLanguage(BaseLanguage):
    file_extension = "py"
    name = "python"

    # Defaults for instances created without __init__ (e.g. in tests that skip
    # kernel startup). __init__ overrides them.
    interpreter = None
    imported_modules = set()
    function_fingerprints = {}
    variable_fingerprints = {}

    def __init__(self, interpreter):
        self.interpreter = interpreter

        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        while not self.kc.is_alive():
            time.sleep(0.1)
        time.sleep(0.5)

        self.listener_thread = None
        self.finish_flag = False

        # Modules known to be bound in the kernel's user namespace. Populated
        # from each block's imports and refreshed from the REPL-state line the
        # kernel reports after every run, so redundant top-level `import X`
        # lines can be stripped before execution.
        self.imported_modules = set()

        # Fingerprints of top-level user definitions already bound in the
        # kernel, refreshed wholesale from the hidden `##oi_fp##` marker after
        # every run. Functions are keyed by their normalized-source fingerprint,
        # scalar variables by their repr fingerprint — so a redundant identical
        # `def f` / `x = 5` can be stripped before execution.
        self.function_fingerprints = {}
        self.variable_fingerprints = {}

        # Use Inline by default for broad compatibility. Users can opt into a GUI backend
        # (e.g. TkAgg/QtAgg) by setting INTERPRETER_MPL_BACKEND or MPLBACKEND.
        # INTERPRETER_MPL_BACKEND takes precedence so Open Interpreter can control behavior.
        configured_backend = os.environ.get("INTERPRETER_MPL_BACKEND", "").strip()
        backend_source = "INTERPRETER_MPL_BACKEND"
        if not configured_backend:
            configured_backend = os.environ.get("MPLBACKEND", "").strip()
            backend_source = "MPLBACKEND"

        if configured_backend:
            code = f"""
import matplotlib
matplotlib.use({configured_backend!r})
import matplotlib.pyplot as plt
_oi_mpl_backend_hint_shown = False
_oi_mpl_original_show = plt.show
def _oi_mpl_show_with_hint(*args, **kwargs):
    global _oi_mpl_backend_hint_shown
    if not _oi_mpl_backend_hint_shown:
        print("Matplotlib backend set by {backend_source}={configured_backend!r}.")
        _oi_mpl_backend_hint_shown = True
    # GUI backends (e.g. TkAgg) can block the OI loop on show(); default to
    # non-blocking so users can interact with the figure while code execution continues.
    kwargs.setdefault("block", False)
    return _oi_mpl_original_show(*args, **kwargs)
plt.show = _oi_mpl_show_with_hint
""".strip()
        else:
            code = """
%matplotlib inline
import matplotlib.pyplot as plt
_oi_mpl_backend_hint_shown = False
_oi_mpl_original_show = plt.show
def _oi_mpl_show_with_hint(*args, **kwargs):
    global _oi_mpl_backend_hint_shown
    if not _oi_mpl_backend_hint_shown:
        print("Matplotlib backend not set; defaulting to inline. Set INTERPRETER_MPL_BACKEND=TkAgg (or MPLBACKEND=TkAgg) for interactive GUI plots (pan/zoom).")
        _oi_mpl_backend_hint_shown = True
    return _oi_mpl_original_show(*args, **kwargs)
plt.show = _oi_mpl_show_with_hint
""".strip()

        for _ in self.run(code):
            pass

        # Configure IPython display formatter to prefer markdown or plain text
        # so that pandas doesn't output dataframes as HTML tables (which are
        # then themselves executed).
        ipython_config = """
from IPython import get_ipython
ip = get_ipython()
ip.display_formatter.active_types = ['text/markdown', 'text/plain']
""".strip()

        for _ in self.run(ipython_config):
            pass

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

    def run(self, code):
        while not self.kc.is_alive():
            time.sleep(0.1)

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

        # if self.toolbox.save_skills and functions:
        #     skill_library_path = self.toolbox.skills.path

        #     if not os.path.exists(skill_library_path):
        #         os.makedirs(skill_library_path)

        #     for filename, function_code in functions.items():
        #         with open(f"{skill_library_path}/{filename}.py", "w") as file:
        #             file.write(function_code)

        self.finish_flag = False
        try:
            try:
                preprocessed_code = self.preprocess_code(code)
            except:
                # Any errors produced here are our fault.
                # Also, for python, you don't need them! It's just for active_line and stuff. Just looks pretty.
                preprocessed_code = code
            message_queue = queue.Queue()
            self._execute_code(preprocessed_code, message_queue)
            for output in self._capture_output(message_queue):
                self._maybe_update_imported_modules(output)
                yield output

            if getattr(self, "kc", None) and self.kc.is_alive():
                for output in self._get_active_state():
                    self._maybe_update_imported_modules(output)
                    yield output
        except GeneratorExit:
            raise  # gotta pass this up!
        except KeyboardInterrupt:
            # Properly handle KeyboardInterrupt: interrupt kernel, clear queue, set finish flag
            self.finish_flag = True
            try:
                self.km.interrupt_kernel()
            except:
                pass
            # Clear any remaining messages from the queue to prevent output sync issues
            while not message_queue.empty():
                try:
                    message_queue.get_nowait()
                except queue.Empty:
                    break
            yield {"type": "console", "format": "output", "content": "KeyboardInterrupt\n"}
        except:
            content = traceback.format_exc()
            yield {"type": "console", "format": "output", "content": content}

    def _execute_code(self, code, message_queue):
        def iopub_message_listener():
            max_retries = 100
            while True:
                # If self.finish_flag = True, and we didn't set it (we do below), we need to stop. That's our "stop"
                if self.finish_flag == True:
                    if DEBUG_MODE:
                        print("interrupting kernel!!!!!")
                    self.km.interrupt_kernel()
                    return
                # For async usage
                if (
                    hasattr(self.interpreter, "stop_event")
                    and self.interpreter.stop_event.is_set()
                ):
                    self.km.interrupt_kernel()
                    self.finish_flag = True
                    return
                try:
                    input_patience = int(
                        os.environ.get("INTERPRETER_TERMINAL_INPUT_PATIENCE", 15)
                    )
                    if (
                        time.time() - self.last_output_time > input_patience
                        and time.time() - self.last_output_message_time > input_patience
                    ):
                        self.last_output_message_time = time.time()

                        text = f"{self.interpreter.messages}\n\nThe program above has been running for over 15 seconds. It might require user input. Are there keystrokes that the user should type in, to proceed after the last command?"

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
                            "model": self.interpreter.llm.model,
                            "stream": True,
                            "temperature": 0,
                        }
                        if self.interpreter.llm.api_key:
                            params["api_key"] = self.interpreter.llm.api_key
                        # Use the same provider endpoint config as the active interpreter model.
                        # Without this, routed models (e.g., dashscope-us/* rewritten as openai/*)
                        # can accidentally call OpenAI with a non-OpenAI key in this side-channel.
                        if self.interpreter.llm.api_base:
                            params["api_base"] = self.interpreter.llm.api_base
                        if self.interpreter.llm.api_version:
                            params["api_version"] = self.interpreter.llm.api_version

                        response = ""
                        for chunk in litellm.completion(**params):
                            content = chunk.choices[0].delta.content
                            if type(content) == str:
                                response += content

                        # Parse the response for input tags
                        input_match = re.search(r"<input>(.*?)</input>", response)
                        if input_match:
                            user_input = input_match.group(1)
                            # Do not automatically send CTRL-C - only send user-provided input
                            if user_input.upper() != "CTRL-C":
                                self.kc.input(user_input)

                    msg = self.kc.iopub_channel.get_msg(timeout=0.05)
                    self.last_output_time = time.time()
                except queue.Empty:
                    continue
                except Exception as e:
                    max_retries -= 1
                    if max_retries < 0:
                        raise
                    print("Jupyter error, retrying:", str(e))
                    continue

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

                content = msg["content"]

                if msg["msg_type"] == "stream":
                    line, active_line = self.detect_active_line(content["text"])
                    active_line_enabled = (
                        os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower()
                        == "true"
                    )
                    if active_line and active_line_enabled:
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
                    data = content["data"]
                    if "image/png" in data:
                        message_queue.put(
                            {
                                "type": "image",
                                "format": "base64.png",
                                "content": data["image/png"],
                            }
                        )
                    elif "image/jpeg" in data:
                        message_queue.put(
                            {
                                "type": "image",
                                "format": "base64.jpeg",
                                "content": data["image/jpeg"],
                            }
                        )
                    elif "text/html" in data:
                        message_queue.put(
                            {
                                "type": "code",
                                "format": "html",
                                "content": data["text/html"],
                            }
                        )
                    elif "text/plain" in data:
                        message_queue.put(
                            {
                                "type": "console",
                                "format": "output",
                                "content": data["text/plain"],
                            }
                        )
                    elif "application/javascript" in data:
                        message_queue.put(
                            {
                                "type": "code",
                                "format": "javascript",
                                "content": data["application/javascript"],
                            }
                        )

        self.listener_thread = threading.Thread(target=iopub_message_listener)
        # self.listener_thread.daemon = True
        self.listener_thread.start()

        if DEBUG_MODE:
            print(
                "thread is on:", self.listener_thread.is_alive(), self.listener_thread
            )

        self.kc.execute(code)

    def detect_active_line(self, line):
        if "##active_line" in line:
            # Split the line by "##active_line" and grab the last element
            last_active_line = line.split("##active_line")[-1]
            # Split the last active line by "##" and grab the first element
            try:
                active_line = int(last_active_line.split("##")[0])
            except:
                active_line = 0
            # Remove all ##active_line{number}##\n
            line = re.sub(r"##active_line\d+##\n", "", line)
            return line, active_line
        return line, None

    def _capture_output(self, message_queue):
        while True:
            time.sleep(0.1)

            # For async usage
            if (
                hasattr(self.interpreter, "stop_event")
                and self.interpreter.stop_event.is_set()
            ):
                self.finish_flag = True
                break

            if self.listener_thread:
                try:
                    output = message_queue.get(timeout=0.1)
                    if DEBUG_MODE:
                        print(output)
                    yield output

                except queue.Empty:
                    if self.finish_flag:
                        time.sleep(0.1)

                        try:
                            output = message_queue.get(timeout=0.1)
                            if DEBUG_MODE:
                                print(output)
                            yield output
                        except queue.Empty:
                            if DEBUG_MODE:
                                print("we're done")
                            break
                except KeyboardInterrupt:
                    # Handle KeyboardInterrupt during output capture
                    self.finish_flag = True
                    # Clear any remaining messages to prevent sync issues
                    while not message_queue.empty():
                        try:
                            message_queue.get_nowait()
                        except queue.Empty:
                            break
                    break

    def stop(self):
        self.finish_flag = True

    def _get_active_state(self):
        state_code = """
import types as __oi_types
import os as __oi_os
__oi_globals = globals()
__oi_cwd = __oi_os.getcwd()
__oi_exclude = ['In', 'Out', 'get_ipython', 'exit', 'quit', 'open', 'original_ps1', 'is_wsl', 'REPLHooks', 'get_last_command', 'PS1', 'ip', 'plt']
__oi_mods = []
__oi_funcs = []
__oi_vars = []

for __oi_k, __oi_v in __oi_globals.items():
    if __oi_k.startswith('_') or __oi_k in __oi_exclude:
        continue
    if isinstance(__oi_v, __oi_types.ModuleType):
        __oi_mods.append(__oi_k)
    elif callable(__oi_v):
        __oi_funcs.append(__oi_k)
    else:
        __oi_vars.append(__oi_k)

__oi_parts = [f"CWD: {__oi_cwd}"]
if __oi_mods:
    __oi_parts.append(f"Already imported: {', '.join(__oi_mods)}")
if __oi_vars:
    __oi_parts.append(f"Variables: {', '.join(__oi_vars)}")
if __oi_funcs:
    __oi_parts.append(f"Functions/Classes: {', '.join(__oi_funcs)}")

# Fingerprint top-level user definitions so the client can strip a
# re-definition that is byte-for-byte identical (same normalized AST) to one
# already bound in the kernel. Functions/classes are fingerprinted from their
# source; scalar variables (immutables the client can literal_eval) from their
# repr. Anything un-fingerprintable is simply omitted — the client then never
# strips that name, which is the safe default. Caps bound the cost of running
# this after every cell.
import ast as __oi_ast
import inspect as __oi_inspect
import hashlib as __oi_hashlib
__oi_fp_parts = []
__oi_fn_n = 0
for __oi_k in __oi_funcs:
    if __oi_fn_n >= 50:
        break
    __oi_o = __oi_globals[__oi_k]
    try:
        __oi_src = __oi_inspect.getsource(__oi_o)
        if len(__oi_src) > 20000:
            continue
        # The executed code has `print('##active_lineN##')` markers injected by
        # the client's preprocessor; they must not be part of the fingerprint
        # (the client fingerprints the raw definition text).
        import re as __oi_re
        __oi_src = __oi_re.sub(
            r"^\s*print\('##active_line\d+##'\)\s*$", "", __oi_src, flags=__oi_re.M
        )
        __oi_fp_parts.append(
            __oi_k + '=fn:' + __oi_hashlib.sha1(
                __oi_ast.dump(__oi_ast.parse(__oi_src).body[0]).encode()
            ).hexdigest()
        )
        __oi_fn_n += 1
    except Exception:
        pass
__oi_var_n = 0
for __oi_k in __oi_vars:
    if __oi_var_n >= 50:
        break
    __oi_v = __oi_globals[__oi_k]
    if not (__oi_v is None or isinstance(__oi_v, (bool, int, float, complex, str, bytes))):
        continue
    __oi_r = repr(__oi_v)
    if len(__oi_r) > 200:
        continue
    __oi_fp_parts.append(
        __oi_k + '=var:' + __oi_hashlib.sha1(__oi_r.encode()).hexdigest()
    )
    __oi_var_n += 1
if __oi_fp_parts:
    print('##oi_fp##' + ','.join(__oi_fp_parts))

__oi_res = f"\\n[Python REPL State: {' | '.join(__oi_parts)}]"
print(__oi_res)
"""
        message_queue = queue.Queue()
        self.finish_flag = False
        self._execute_code(state_code.strip(), message_queue)

        for output in self._capture_output(message_queue):
            if output.get("type") == "console" and output.get("format") == "output":
                # The marker and the REPL-state line can arrive in the same
                # stream chunk, so handle them line by line: parse the hidden
                # fingerprint marker, show everything else.
                for line in output.get("content").split("\n"):
                    if line.startswith("##oi_fp##"):
                        self._update_fingerprints(line)
                    elif line:
                        yield {**output, "content": line}
                continue
            yield output

    _STATE_MODULES_RE = re.compile(r"Already imported:\s*([^|\]]*)")

    def _update_fingerprints(self, marker):
        """Parse the hidden ``##oi_fp##`` marker into the tracked fingerprint dicts.

        The kernel emits one marker per run with ``name=fn:<sha1>`` (function
        source fingerprint) and ``name=var:<sha1>`` (scalar repr fingerprint)
        entries, comma-separated. Adopted wholesale each run so the client's
        view is authoritative and always matches the kernel's live namespace —
        e.g. a later cell that rebinds ``f = other`` updates f's fingerprint
        and a stale ``def f`` is no longer stripped.
        """
        fn_fps, var_fps = {}, {}
        for token in marker[len("##oi_fp##") :].split(","):
            name, _, value = token.partition("=")
            if not name or not value:
                continue
            if value.startswith("fn:"):
                fn_fps[name] = value[len("fn:") :]
            elif value.startswith("var:"):
                var_fps[name] = value[len("var:") :]
        self.function_fingerprints = fn_fps
        self.variable_fingerprints = var_fps

    def _maybe_update_imported_modules(self, output):
        """Refresh the tracked module set from the kernel's REPL-state line.

        The kernel reports exactly which modules are bound in its user
        namespace after every run, so when that line appears we adopt it
        wholesale — this corrects optimistic entries recorded from blocks that
        failed to execute (e.g. an `import sklearn` that raised).
        """
        if not isinstance(output, dict):
            return
        content = output.get("content")
        if not isinstance(content, str):
            return
        m = self._STATE_MODULES_RE.search(content)
        if not m:
            return
        modules = [name.strip() for name in m.group(1).split(",") if name.strip()]
        if modules:
            self.imported_modules = set(modules)

    def strip_boilerplate(self, code):
        """Return (stripped_code, notice) after removing redundant top-level boilerplate.

        Two passes:
        - ``strip_redundant_imports`` drops plain ``import X`` lines for
          allowlisted boilerplate modules the kernel reports as already bound.
          It deliberately does NOT learn new imports from the code it is
          handed: recording ``import time`` here would make a second
          ``strip_boilerplate`` call (e.g. from ``preprocess_code`` during the
          same execution) strip that import before it ever ran, breaking code
          that genuinely needs it. Only the kernel's authoritative REPL-state
          line drives ``imported_modules``.
        - ``strip_redundant_definitions_and_assignments`` drops top-level
          ``def``/``async def`` and scalar assignments (``x = 5``) that
          re-define, byte-for-byte identical, an object already bound in the
          kernel (matched by fingerprint). A redefinition with different code
          is always kept.
        """
        if not getattr(self.interpreter, "strip_redundant_code", True):
            return code, None
        stripped, removed = strip_redundant_imports(code, self.imported_modules)
        notices = []
        if removed:
            distinct = sorted(set(removed))[:4]
            label = "imports" if len(set(removed)) > 1 else "import"
            notices.append(
                f"Removed redundant {label} {', '.join(distinct)} (already imported)."
            )
        stripped, defs_removed, vars_removed = strip_redundant_definitions_and_assignments(
            stripped, self.function_fingerprints, self.variable_fingerprints
        )
        if defs_removed:
            distinct = sorted(set(defs_removed))[:4]
            label = "definitions of" if len(defs_removed) > 1 else "definition of"
            notices.append(
                f"Removed redundant {label} {', '.join(distinct)} (already defined identically)."
            )
        if vars_removed:
            distinct = sorted(set(vars_removed))[:4]
            label = "assignments to" if len(vars_removed) > 1 else "assignment to"
            notices.append(
                f"Removed redundant {label} {', '.join(distinct)} (already set to that value)."
            )
        if notices:
            return stripped, "; ".join(notices)
        return stripped, None

    def preprocess_code(self, code):
        code, _ = self.strip_boilerplate(code)
        return preprocess_python(code)


# Modules we're willing to drop a redundant top-level `import X` for. Only
# side-effect-free imports that LLMs habitually repeat at the top of every cell
# are listed. Deliberately NOT included: modules whose import has side effects
# (e.g. matplotlib picks a backend), and the near-universal aliased imports
# (`import numpy as np`, `import pandas as pd`, `import matplotlib.pyplot as plt`)
# — those have aliases/dots and are never stripped anyway. Stripping also
# requires the module to already be bound in the kernel, so a first import
# (with any real side effect) is never affected.
REMOVABLE_BOILERPLATE_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "re",
        "json",
        "time",
        "datetime",
        "random",
        "math",
        "subprocess",
        "glob",
        "shutil",
        "pathlib",
        "string",
        "typing",
        "tempfile",
        "uuid",
        "hashlib",
        "base64",
        "io",
        "csv",
        "statistics",
        "secrets",
        "pprint",
        "copy",
        "warnings",
        "platform",
        "textwrap",
        "collections",
        "functools",
        "itertools",
        "logging",
        "argparse",
        "heapq",
        "bisect",
        "decimal",
        "fractions",
        "operator",
        "queue",
        "threading",
        "struct",
        "zlib",
        "gzip",
        "pickle",
        "sqlite3",
        "socket",
        "requests",
    }
)


def strip_redundant_imports(code, imported_modules):
    """Drop top-level ``import X`` lines for boilerplate modules already bound.

    Only plain, single-line, column-0 imports without aliases or dotted names
    are considered (``import os``, ``import os, sys``), and only while they sit
    in the *leading* block — before the first executable statement (comments
    and blank lines may precede them). An ``import`` after any other statement
    is never stripped: it may re-bind a name that was assigned earlier (e.g.
    ``os = "string"`` followed by ``import os``) or be an intentional mid-cell
    import, so removing it could break the code. The module must also be in
    REMOVABLE_BOILERPLATE_IMPORTS: a deliberately tiny allowlist of
    side-effect-free stdlib modules that LLMs habitually re-import and that the
    kernel already binds at startup. Anything else — matplotlib and other
    imports with side effects, aliased/dotted/``from X import y`` lines, and
    indented function-scope imports — is always left alone.

    Returns ``(stripped_code, removed_module_names)`` so the caller can notify
    the user about what was dropped.
    """
    removed = []
    if not imported_modules:
        return code, removed
    kept_lines = []
    in_leading = True
    for line in code.splitlines():
        if line.strip() == "" or line.lstrip().startswith("#"):
            kept_lines.append(line)  # blank/comment — stay in the leading block
            continue
        if in_leading:
            m = re.match(r"^import\s+(.+)$", line)
            if m:
                names = [name.split("#")[0].strip() for name in m.group(1).split(",")]
                if not names or any(" as " in name or "." in name for name in names):
                    kept_lines.append(line)
                    continue
                if all(
                    name in imported_modules
                    and name in REMOVABLE_BOILERPLATE_IMPORTS
                    for name in names
                ):
                    removed.extend(names)  # redundant boilerplate — drop the line
                    continue
                kept_lines.append(line)
                continue
            if re.match(r"^from\s+\S+\s+import\s", line):
                kept_lines.append(line)  # never stripped, but still part of the leading block
                continue
            in_leading = False  # first executable statement ends the leading block
        kept_lines.append(line)
    return "\n".join(kept_lines), removed


def strip_redundant_definitions_and_assignments(code, function_fps, variable_fps):
    """Drop top-level definitions/scalar assignments already bound identically.

    Removes a top-level ``def``/``async def`` whose normalized source
    fingerprint matches one the kernel already binds to that name, and a
    single-name scalar assignment (``x = 5``, ``x = "abc"``, ``x = None``)
    whose repr fingerprint matches. The fingerprint match means the re-run is a
    no-op, so removing it cannot change behavior. A redefinition with different
    code always survives.

    Safety rules:
    - Only *top-level* statements are considered; anything inside a function/
      class/block is untouched.
    - A statement is stripped only if its name is not bound earlier in the
      cell (``def f`` then a *different* ``def f`` keeps both; ``x = 6`` then
      ``x = 5`` keeps both — stripping the second would change the result).
      Kept statements mark the names they bind (assigns, defs, imports,
      for/with targets, walrus, ``del``), so a later same-name candidate is
      conservatively kept.
    - Mutable objects, non-literal RHS (``2 + 3``, ``int("5")``, ``f(x)``) and
      anything the kernel couldn't fingerprint are never stripped (the kernel
      only fingerprints immutable scalars, so ``x = [1, 2]`` has no matching
      entry and is left alone — re-assigning a fresh list isn't a no-op).
    - The span of each removed statement is re-parsed after removal; if that
      produces invalid Python (e.g. a shared ``x = 5; y = 6`` line), the
      statement is kept. Compound statements never share a ``;`` line, so defs
      have no such seam.

    Returns ``(stripped_code, removed_function_names, removed_var_names)``.
    """
    removed_funcs, removed_vars = [], []
    if not function_fps and not variable_fps:
        return code, removed_funcs, removed_vars

    work = code
    bound = set()
    while True:
        try:
            tree = ast.parse(work)
        except SyntaxError:
            break  # magics / `!cmd` / incomplete code — never touch it
        if not tree.body:
            break
        for stmt in tree.body:
            span, kind = _redundant_span(stmt, work, function_fps, variable_fps, bound)
            if span is None:
                bound |= _bound_names(stmt)
                continue
            # Try the removal; re-parse to verify it doesn't break the cell.
            candidate = work[: span[0]] + work[span[1] :]
            try:
                ast.parse(candidate)
            except SyntaxError:
                bound |= _bound_names(stmt)  # unsafe — keep it, treat as bound
                continue
            work = candidate
            if kind == "fn":
                removed_funcs.append(stmt.name)
            else:
                removed_vars.append(stmt.targets[0].id)
            break  # offsets shifted — restart the walk on the new text
        else:
            break  # no candidate removals in this pass
    return work, removed_funcs, removed_vars


def _bound_names(stmt):
    """Names a top-level statement binds, for in-cell redundancy tracking.

    Conservative: anything assigned anywhere in the statement's subtree counts,
    even inside nested function bodies — over-marking only suppresses stripping,
    never causes a wrong strip.
    """
    names = set()
    for node in ast.walk(stmt):
        # `del x` uses Del ctx (not Store); either way the name's prior
        # binding no longer holds after the statement, so a later identical
        # reassign must be kept.
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(stmt.name)
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        for alias in stmt.names:
            names.add(alias.asname or alias.name.split(".")[0])
    return names


def _redundant_span(stmt, source, function_fps, variable_fps, bound):
    """Return (span, kind) if ``stmt`` is a redundant definition, else (None, None).

    ``span`` is (start_offset, end_offset) into ``source``. ``kind`` is "fn"
    for functions or "var" for scalar assignments.
    """
    start_line = stmt.lineno
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if stmt.name not in function_fps or stmt.name in bound:
            return None, None
        try:
            fp = _function_fingerprint(stmt)
        except Exception:
            return None, None
        if fp != function_fps[stmt.name]:
            return None, None
        if stmt.decorator_list:
            start_line = min(d.lineno for d in stmt.decorator_list)
        return _span_offsets(source, start_line, 0, stmt.end_lineno, stmt.end_col_offset), "fn"

    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(
        stmt.targets[0], ast.Name
    ):
        name = stmt.targets[0].id
        if name not in variable_fps or name in bound:
            return None, None
        try:
            value = ast.literal_eval(stmt.value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None, None  # not a literal — can't be a no-op re-assign
        if _value_fingerprint(value) != variable_fps[name]:
            return None, None
        start, end = _span_offsets(
            source, stmt.lineno, stmt.col_offset, stmt.end_lineno, stmt.end_col_offset
        )
        # Only strip an assignment that occupies its own line. A shared line
        # (`x = 5; y = 6`) leaves a dangling `;` or a now-undefined sibling if
        # one target is removed, and re-parsing can't catch the semantics, so
        # require nothing non-whitespace before or after the span on its line.
        line_start = _span_offsets(source, stmt.lineno, 0, stmt.lineno, 0)[0]
        line_tail = source[end:].split("\n", 1)[0]
        if source[line_start:start].strip() or line_tail.strip():
            return None, None
        return (start, end), "var"

    return None, None


def _span_offsets(source, start_line, start_col, end_line, end_col):
    """Convert line/col positions to absolute string offsets."""
    offsets = [0]
    for line in source.split("\n"):
        offsets.append(offsets[-1] + len(line) + 1)  # +1 for the newline
    start = offsets[start_line - 1] + start_col
    end = offsets[end_line - 1] + end_col
    return start, end


def _function_fingerprint(stmt):
    """Normalized source fingerprint of a FunctionDef/AsyncFunctionDef node.

    Uses ``ast.dump`` of the node (structure only, no line numbers or
    formatting), matching the kernel's ``ast.dump(ast.parse(src).body[0])``.
    """
    return hashlib.sha1(ast.dump(stmt).encode()).hexdigest()


def _value_fingerprint(value):
    """Repr fingerprint of an immutable scalar, matching the kernel's ``var:`` entries."""
    return hashlib.sha1(repr(value).encode()).hexdigest()


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
    except:
        # If you can't parse the processed version, try the unprocessed version before giving up
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
            docstring = ast.get_docstring(node)
            body = node.body
            if docstring:
                body = body[1:]

            code_body = ast.unparse(body[0]).replace("\n", "\n    ")

            func_info = {
                "name": node.name,
                "docstring": docstring,
                "body": code_body,
            }
            functions.append(func_info)

    for func in functions:
        # Consolidating import statements and function definition
        function_content = "\n".join(import_statements) + "\n\n"
        function_content += f"def {func['name']}():\n    \"\"\"{func['docstring']}\"\"\"\n    {func['body']}\n"

        # Adding to dictionary
        functions_dict[func["name"]] = function_content

    return functions_dict
