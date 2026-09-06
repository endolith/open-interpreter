import sys

from rich import print as rich_print


def prompt_choice(prompt, choices):
    """
    Prompt until the user enters one of the given single-character choices.
    Returns the choice. choices e.g. ('y', 'n') or ('y', 'a', 'n').

    The full prompt is shown once. On invalid input, only the hint is printed
    (with Rich so choices appear in bold) and a minimal reprompt, no extra newlines.
    """
    choices = tuple(c.lower() for c in choices)
    if len(choices) <= 1:
        hint = "Please press " + "".join(f"[bold]{c}[/bold]" for c in choices) + "."
    elif len(choices) == 2:
        hint = "Please press [bold]" + choices[0] + "[/bold] or [bold]" + choices[1] + "[/bold]."
    else:
        hint = "Please press " + ", ".join(f"[bold]{c}[/bold]" for c in choices[:-1]) + ", or [bold]" + choices[-1] + "[/bold]."
    reprompt = "  "
    current_prompt = prompt
    # Non-interactive (server/headless) mode: no TTY to read from. Rather than
    # crash with EOFError, take the last choice, which is the safe/decline option
    # by convention here ('n' in y/n, 'n' in y/a/n).
    if not (sys.stdin and sys.stdin.isatty()):
        return choices[-1]
    while True:
        try:
            response = input(current_prompt).strip().lower()
        except EOFError:
            return choices[-1]
        response = response[:1] if response else ""
        if response in choices:
            print("")
            return response
        rich_print(hint)
        current_prompt = reprompt
