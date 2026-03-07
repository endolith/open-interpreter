from rich import print as rich_print
from rich.markdown import Markdown
from rich.padding import Padding


def display_markdown_message(message):
    """
    Display markdown message. Renders the full message as Markdown so tables and
    other multi-line constructs are rendered correctly.
    """
    if "traceback" in message.lower() or "error" in message.lower():
        print(message)
        return

    rich_print(Padding(Markdown(message), (1, 2, 0, 2)))
    if "\n" not in message and message.startswith(">"):
        print("")
