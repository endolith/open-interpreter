import json
import os
import platform
import time
import subprocess
import getpass

from ..utils.recipient_utils import parse_for_recipient
from .languages.applescript import AppleScript
from .languages.bash import Bash
from .languages.cmd import Cmd
from .languages.html import HTML
from .languages.java import Java
from .languages.javascript import JavaScript
from .languages.powershell import PowerShell
from .languages.python import Python
from .languages.r import R
from .languages.react import React
from .languages.ruby import Ruby
from .languages.perl import Perl
from .languages.augeas import Augeas

# Languages whose console output is buffered until completion (reduces UI flicker).
_BUFFERED_CONSOLE_LANGUAGES = frozenset({"cmd", "bash"})


def _sync_active_line_detection_env(interpreter):
    """
    Keep INTERPRETER_ACTIVE_LINE_DETECTION in sync with interpreter.highlight_active_line.

    highlight_active_line in default.yaml controls both:
      - UI highlighting in code blocks (CodeBlock)
      - Whether preprocessors inject ##active_line## markers into executed code
    """
    enabled = True
    if hasattr(interpreter, "highlight_active_line") and interpreter.highlight_active_line is not None:
        enabled = bool(interpreter.highlight_active_line)
    os.environ["INTERPRETER_ACTIVE_LINE_DETECTION"] = "true" if enabled else "false"


def _default_terminal_languages():
    languages = [
        Ruby,
        Python,
        Bash,
        JavaScript,
        HTML,
        AppleScript,
        R,
        PowerShell,
        React,
        Java,
        Perl,
        Augeas,
    ]
    if platform.system() == "Windows":
        languages.insert(3, Cmd)
    return languages


class Terminal:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.languages = _default_terminal_languages()
        self._active_languages = {}

    def sudo_install(self, package):
        try:
            # First, try to install without sudo
            subprocess.run(['apt', 'install', '-y', package], check=True)
        except subprocess.CalledProcessError:
            # If it fails, try with sudo
            print(f"Installation of {package} requires sudo privileges.")
            sudo_password = getpass.getpass("Enter sudo password: ")

            try:
                # Use sudo with password
                subprocess.run(
                    ['sudo', '-S', 'apt', 'install', '-y', package],
                    input=sudo_password.encode(),
                    check=True
                )
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}. Error: {e}")
                return False

        return True

    def _build_import_toolbox_api_code(self):
        """
        Build the snippet that imports the Toolbox API into the kernel.

        The kernel's working directory is the directory the CLI was started
        from, and the kernel puts that directory on its sys.path. If the CLI
        is launched from inside a *different* open-interpreter checkout (for
        example the `main` branch repo, whose `interpreter` package has a
        `computer` object rather than `toolbox` and no `ai2`), the kernel's
        `from interpreter import interpreter, ai2` would import that wrong
        copy and the injection would fail, leaving `toolbox` undefined.

        Prepending the parent process's own interpreter package directory to
        sys.path makes the kernel import the exact same package the CLI is
        running, regardless of what folder it was started from.

        The trailing print is a sentinel: it lets the caller detect whether
        the injection actually succeeded before marking it as done.
        """
        import importlib
        interpreter_module = importlib.import_module("interpreter")
        interpreter_pkg_dir = os.path.dirname(
            os.path.dirname(interpreter_module.__file__)
        )
        return f'''
import os
os.environ["INTERPRETER_TOOLBOX_API"] = "False" # To prevent infinite recurring injection of the toolbox API
import sys
_pkg_dir = {interpreter_pkg_dir!r}
# Always re-insert: the package dir may already be on sys.path (e.g. via an
# editable install's .pth), but the kernel's working directory — or anything
# else — may precede it, and the FIRST matching entry wins for imports.
sys.path.insert(0, _pkg_dir)
import time
import datetime
from interpreter import interpreter, ai2

toolbox = interpreter.toolbox
print("__TOOLBOX_API_IMPORTED__")
'''.strip()

    def get_language(self, language):
        for lang in self.languages:
            if language.lower() == lang.name.lower():
                return lang
        return None

    def get_language_instance(self, language):
        """Return the live language instance for `language`, or None if not created yet.

        Only instances already created by run/_streaming_run are returned.
        Instantiating here could eagerly boot a Python kernel or probe for an
        executable, so callers should only inspect (not create) state. There is
        no tracking state to strip against before a language's first run anyway.
        """
        return self._active_languages.get(language)

    def run(self, language, code, stream=False, display=False):
        # Check if this is an apt install command
        if language == "bash" and code.strip().startswith("apt install"):
            package = code.split()[-1]
            if self.sudo_install(package):
                return [{"type": "console", "format": "output", "content": f"Package {package} installed successfully."}]
            else:
                return [{"type": "console", "format": "output", "content": f"Failed to install package {package}."}]

        if language == "python":
            if (
                self.interpreter.toolbox.import_toolbox_api
                and not self.interpreter.toolbox._has_imported_toolbox_api
                and ("toolbox" in code or "ai2" in code)
                and os.getenv("INTERPRETER_TOOLBOX_API", "True") != "False"
            ):
                # Mark as imported *before* running the injection, so the
                # injection's own recursive `self.run` (its code contains
                # "toolbox") doesn't re-trigger itself. If the injection fails
                # (e.g. the kernel imported the wrong copy of the package),
                # restore the flag so the next toolbox use retries instead of
                # silently leaving `toolbox` undefined, which would surface
                # later as a confusing `NameError: name 'toolbox' is not defined`.
                self.interpreter.toolbox._has_imported_toolbox_api = True
                # Give it access to the toolbox via Python
                time.sleep(0.5)
                results = self.run(
                    language="python",
                    code=self._build_import_toolbox_api_code(),
                    display=self.interpreter.toolbox.verbose,
                )
                if not any(
                    "__TOOLBOX_API_IMPORTED__" in chunk.get("content", "")
                    for chunk in results
                    if chunk.get("format") == "output"
                ):
                    self.interpreter.toolbox._has_imported_toolbox_api = False

            if self.interpreter.toolbox.import_skills and not self.interpreter.toolbox._has_imported_skills:
                self.interpreter.toolbox._has_imported_skills = True
                self.interpreter.toolbox.skills.import_skills()

            # This won't work because truncated code is stored in interpreter.messages :/
            # If the full code was stored, we could do this:
            if False and "get_last_output()" in code:
                if "# We wouldn't want to have maximum recursion depth!" in code:
                    # We just tried to run this, in a moment.
                    pass
                else:
                    code_outputs = [
                        m
                        for m in self.interpreter.messages
                        if m["role"] == "computer"
                        and "content" in m
                        and m["content"] != ""
                    ]
                    if len(code_outputs) > 0:
                        last_output = code_outputs[-1]["content"]
                    else:
                        last_output = ""
                    last_output = json.dumps(last_output)

                    self.run(
                        "python",
                        f"# We wouldn't want to have maximum recursion depth!\nimport json\ndef get_last_output():\n    return '''{last_output}'''",
                    )

        if stream == False:
            # If stream == False, *pull* from _streaming_run.
            output_messages = []
            for chunk in self._streaming_run(language, code, display=display):
                if chunk.get("format") != "active_line":
                    # Should we append this to the last message, or make a new one?
                    if (
                        output_messages != []
                        and output_messages[-1].get("type") == chunk["type"]
                        and output_messages[-1].get("format") == chunk["format"]
                    ):
                        output_messages[-1]["content"] += chunk["content"]
                    else:
                        output_messages.append(chunk)
            return output_messages

        elif stream == True:
            # If stream == True, replace this with _streaming_run.
            return self._streaming_run(language, code, display=display)

    def _streaming_run(self, language, code, display=False):
        start_time = time.time()
        _sync_active_line_detection_env(self.interpreter)

        if language not in self._active_languages:
            # Get the language. Pass in self.interpreter *if it takes a single argument>
            # but pass in nothing if not. This makes custom languages easier to add / understand.
            lang_class = self.get_language(language)
            if lang_class is None:
                yield {
                    "type": "console",
                    "format": "output",
                    "content": f"Unknown language: {language!r}. Use cmd, bash, or powershell on Windows; bash on Linux/Mac.",
                }
                return
            if lang_class.__init__.__code__.co_argcount > 1:
                self._active_languages[language] = lang_class(self.interpreter)
            else:
                self._active_languages[language] = lang_class()
        try:
            buffered_output = ""

            for chunk in self._active_languages[language].run(code):
                # self.format_to_recipient can format some messages as having a certain recipient.
                # Here we add that to the LMC messages:
                if chunk["type"] == "console" and chunk.get("format") == "output":
                    # Add timing to the end of the output
                    if "end" in chunk:
                        elapsed = round(time.time() - start_time, 2)
                        chunk["content"] = f"{chunk['content'].strip()}\n\nTime elapsed: {elapsed}s"

                    recipient, content = parse_for_recipient(chunk["content"])
                    if recipient:
                        chunk["recipient"] = recipient
                        chunk["content"] = content

                    # Sometimes, we want to hide the traceback to preserve tokens.
                    # (is this a good idea?)
                    if "@@@HIDE_TRACEBACK@@@" in content:
                        chunk["content"] = (
                            "Stopping execution.\n\n"
                            + content.split("@@@HIDE_TRACEBACK@@@")[-1].strip()
                        )

                    if language in _BUFFERED_CONSOLE_LANGUAGES:
                        if not buffered_output:
                            yield {
                                "type": "console",
                                "format": "output",
                                "content": "Note: Shell command output will be shown after completion.\n\n"
                            }
                        buffered_output += chunk["content"]
                        continue

                    yield chunk

                    if (
                        display
                        and chunk.get("format") != "active_line"
                        and chunk.get("content")
                        and language not in _BUFFERED_CONSOLE_LANGUAGES
                    ):
                        print(chunk["content"], end="")

                else:
                    yield chunk

            if buffered_output:
                elapsed = round(time.time() - start_time, 2)
                yield {
                    "type": "console",
                    "format": "output",
                    "content": f"{buffered_output.strip()}\n\nTime elapsed: {elapsed}s"
                }

        except GeneratorExit:
            self.stop()

    def stop(self):
        for language in self._active_languages.values():
            language.stop()

    def terminate(self):
        for language_name in list(self._active_languages.keys()):
            language = self._active_languages[language_name]
            if (
                language
            ):  # Not sure why this is None sometimes. We should look into this
                language.terminate()
            del self._active_languages[language_name]
