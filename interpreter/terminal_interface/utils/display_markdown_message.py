from rich import print as rich_print
from rich.markdown import Markdown
from rich.rule import Rule


def display_markdown_message(message):
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
            except UnicodeEncodeError:
                # The offending line cannot be printed back out verbatim:
                # print() would raise the same error on the same character.
                # Emit the escaped form so the content stays visible to a
                # human reading a bug report without re-raising.
                print(
                    "Error displaying line:",
                    line.encode("unicode_escape").decode("ascii"),
                )

    if "\n" not in message and message.startswith(">"):
        # Aesthetic choice. For these tags, they need a space below them
        print("")
