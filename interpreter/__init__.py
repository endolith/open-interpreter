import sys
import warnings

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

# Skip heavy imports (litellm etc.) for flags that exit immediately — the CLI
# entry point exits via argparse/sys.exit before it ever needs these names.
_FAST_EXIT_FLAGS = {"--help", "-h", "--version"}
if not _FAST_EXIT_FLAGS.intersection(sys.argv):
    from .core.async_core import AsyncInterpreter
    from .core.terminal.base_language import BaseLanguage
    from .core.core import OpenInterpreter

    interpreter = OpenInterpreter()
    toolbox = interpreter.toolbox

    class _LazyAi2Proxy:
        """Expose interpreter.ai2 without constructing Ai2 until first use."""

        def _target(self):
            return interpreter.toolbox.ai2

        def __getattr__(self, name):
            return getattr(self._target(), name)

        def __repr__(self):
            return repr(self._target())

    ai2 = _LazyAi2Proxy()

#     ____                      ____      __                            __
#    / __ \____  ___  ____     /  _/___  / /____  _________  ________  / /____  _____
#   / / / / __ \/ _ \/ __ \    / // __ \/ __/ _ \/ ___/ __ \/ ___/ _ \/ __/ _ \/ ___/
#  / /_/ / /_/ /  __/ / / /  _/ // / / / /_/  __/ /  / /_/ / /  /  __/ /_/  __/ /
#  \____/ .___/\___/_/ /_/  /___/_/ /_/\__/\___/_/  / .___/_/   \___/\__/\___/_/
#      /_/                                         /_/
