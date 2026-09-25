from rich import print as rich_print

from .sanitize_terminal_input import sanitize_terminal_input


def prompt_choice(prompt, choices):
    """
    Prompt until the user enters exactly one of the given choices.
    Returns the choice. choices e.g. ('y', 'n') or ('y', 'a', 'n').

    An answer must be a single letter: a word like "yes" or a run of letters
    like "yqn" is rejected outright instead of being read as its first
    character. Truncating would let a stray keypress inside a paste or a
    multi-key answer act as a deliberate choice (a queued "yes" would silently
    start a code-execution prompt, a telemetry upload, or an infinite retry
    loop), so only an exact match counts and anything else re-prompts.

    Surrounding whitespace and letter case are still forgiving, since those
    carry no other intent: " Y " answers "y".

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
    while True:
        # Strip leaked mouse/report bytes (a poisoned log can arm tracking on
        # replay) so a movement while answering y/n doesn't fail validation.
        # Membership is an exact match, not a prefix match, so a multi-letter
        # answer is rejected rather than silently reduced to its first letter.
        response = sanitize_terminal_input(input(current_prompt)).strip().lower()
        if response in choices:
            print("")
            return response
        rich_print(hint)
        current_prompt = reprompt
