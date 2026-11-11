"""Open Interpreter.

Importing this package sets up environment variables and warning filters but
otherwise has no side effects. The module-level singleton
(`from interpreter import interpreter`) and the public classes are created on
first access via PEP 562, so `import interpreter` stays cheap and never
constructs an interpreter.
"""

import importlib
import os
import sys
import warnings

# OpenRouter (via LiteLLM): optional HTTP-Referer and app title for openrouter.ai rankings.
os.environ.setdefault("OR_SITE_URL", "https://github.com/endolith/open-interpreter")
os.environ.setdefault("OR_APP_NAME", "Open Interpreter")

# Suppress pydantic warning from litellm about fields being removed in V2
warnings.filterwarnings(
    "ignore",
    message="Valid config keys have changed in V2:*",
    module="pydantic.*"
)

if "--os" in sys.argv:
    from rich import print as rich_print
    from rich.markdown import Markdown
    from rich.rule import Rule

    def print_markdown(message):
        """
        Display markdown message. Works with multiline strings with lots of indentation.
        Will automatically make single line > tags beautiful.
        """

        for line in message.split("\n"):
            line = line.strip()
            if line == "":
                print("")
            elif line == "---":
                rich_print(Rule(style="white"))
            else:
                try:
                    rich_print(Markdown(line))
                except UnicodeEncodeError as e:
                    # Replace the problematic character or handle the error as needed
                    print("Error displaying line:", line)

        if "\n" not in message and message.startswith(">"):
            # Aesthetic choice. For these tags, they need a space below them
            print("")

    from importlib.metadata import version
    import requests
    from packaging import version

    def check_for_update():
        # Fetch the latest version from the PyPI API
        response = requests.get(f"https://pypi.org/pypi/open-interpreter/json")
        latest_version = response.json()["info"]["version"]

        # Get the current version using importlib.metadata
        current_version = version("open-interpreter")

        return version.parse(latest_version) > version.parse(current_version)

    if check_for_update():
        print_markdown(
            "> **A new version of Open Interpreter is available.**\n>Please run: `pip install --upgrade open-interpreter`\n\n---"
        )

    if "--voice" in sys.argv:
        print("Coming soon...")
    from .computer_use.loop import run_async_main

    run_async_main()
    exit()

_CLASSES = {
    "OpenInterpreter": ("interpreter.core.core", "OpenInterpreter"),
    "AsyncInterpreter": ("interpreter.core.async_core", "AsyncInterpreter"),
    "BaseLanguage": ("interpreter.core.terminal.base_language", "BaseLanguage"),
}

_singleton = None
_constructing = False


def _get_singleton():
    global _singleton, _constructing
    if _singleton is None:
        if _constructing:
            # OpenInterpreter.__init__ imports modules that may reach back for
            # `interpreter`; without this the recursion would build a second
            # instance and hand out whichever finished last.
            raise RuntimeError(
                "interpreter.interpreter was accessed while it was being constructed"
            )
        _constructing = True
        try:
            module = importlib.import_module("interpreter.core.core")
            _singleton = module.OpenInterpreter()
        finally:
            _constructing = False
    return _singleton


def __getattr__(name):
    if name == "interpreter":
        return _get_singleton()
    if name == "toolbox":
        return _get_singleton().toolbox
    # Toolbox.ai2 is itself a lazy property, so this stays cheap until use.
    if name == "ai2":
        return _get_singleton().toolbox.ai2
    if name in _CLASSES:
        module_name, attr = _CLASSES[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module 'interpreter' has no attribute {name!r}")


__all__ = [
    "interpreter",
    "toolbox",
    "ai2",
    "OpenInterpreter",
    "AsyncInterpreter",
    "BaseLanguage",
]

#     ____                      ____      __                            __
#    / __ \____  ___  ____     /  _/___  / /____  _________  ________  / /____  _____
#   / / / / __ \/ _ \/ __ \    / // __ \/ __/ _ \/ ___/ __ \/ ___/ _ \/ __/ _ \/ ___/
#  / /_/ / /_/ /  __/ / / /  _/ // / / / /_/  __/ /  / /_/ / /  /  __/ /_/  __/ /
#  \____/ .___/\___/_/ /_/  /___/_/ /_/\__/\___/_/  / .___/_/   \___/\__/\___/_/
#      /_/                                         /_/
