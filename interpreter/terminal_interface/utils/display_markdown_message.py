from rich import print as rich_print
from rich.markdown import Markdown
from rich.padding import Padding

from .display_constants import PADDING_MESSAGE


def display_markdown_message(message):
    """
    Display markdown message. Renders the full message as Markdown so tables and
    other multi-line constructs are rendered correctly.
    """
    try:
        rich_print(Padding(Markdown(message), PADDING_MESSAGE))
    except UnicodeEncodeError:
        # The offending line cannot be printed back out verbatim: print()
        # would raise the same error on the same character. Emit the escaped
        # form so the content stays visible to a human reading a bug report.
        for line in message.split("\n"):
            try:
                print(line)
            except UnicodeEncodeError:
                print(
                    "Error displaying line:",
                    line.encode("unicode_escape").decode("ascii"),
                )
    if "\n" not in message and message.startswith(">"):
        print("")
