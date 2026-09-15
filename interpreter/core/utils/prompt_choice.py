import sys

from rich import print as rich_print


class NoInteractiveInput(Exception):
    """
    Raised when a choice is needed but there is no terminal to read it from.

    prompt_choice deliberately does not pick an answer of its own. Only the
    caller knows what "nobody can answer this" should mean: skipping a code run
    lets the model try something else, while skipping the profile migration
    silently throws the user's whole profile away. Each call site handles this.
    """

    def __init__(self, prompt, choices):
        self.prompt = prompt
        self.choices = choices
        super().__init__(
            "No interactive terminal is available to answer: "
            f"{' '.join(prompt.split())} ({'/'.join(choices)})"
        )


def stdin_is_interactive():
    """False for uvicorn workers, containers, pytest, and other environments with no real TTY."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def prompt_choice(prompt, choices):
    """
    Prompt until the user enters one of the given single-character choices.
    Returns the choice. choices e.g. ('y', 'n') or ('y', 'a', 'n').

    The full prompt is shown once. On invalid input, only the hint is printed
    (with Rich so choices appear in bold) and a minimal reprompt, no extra newlines.

    Raises NoInteractiveInput when there is no TTY to read from, rather than
    crashing with EOFError or guessing a choice.
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
    # Non-interactive (server/container/headless): there is no one to ask.
    if not stdin_is_interactive():
        raise NoInteractiveInput(prompt, choices)
    while True:
        try:
            response = input(current_prompt).strip().lower()
        except EOFError:
            # stdin closed mid-session, after isatty() was true at entry.
            raise NoInteractiveInput(prompt, choices) from None
        response = response[:1] if response else ""
        if response in choices:
            print("")
            return response
        rich_print(hint)
        current_prompt = reprompt
