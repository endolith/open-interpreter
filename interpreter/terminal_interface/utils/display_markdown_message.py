from rich import print as rich_print
from rich.markdown import Markdown


def display_markdown_message(message):
    """
    Display markdown message. Renders the full message as Markdown so tables and
    other multi-line constructs are rendered correctly.
    """
    if "traceback" in message.lower() or "error" in message.lower():
        print(message)
        return

    rich_print(Markdown(message))
    if "\n" not in message and message.startswith(">"):
        print("")
