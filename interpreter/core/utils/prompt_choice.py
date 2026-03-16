def prompt_choice(prompt, choices):
    """
    Prompt until the user enters one of the given single-character choices.
    Returns the choice. choices e.g. ('y', 'n') or ('y', 'a', 'n').

    The full prompt is shown once. On invalid input, only the hint is printed
    and a bare reprompt is used so the full prompt text isn't repeated.
    """
    choices = tuple(c.lower() for c in choices)
    hint = "Please enter " + ", ".join(f"'{c}'" for c in choices) + ".\n"
    # Use the trailing whitespace of the original prompt (e.g. "  ") as reprompt
    # so alignment stays consistent without repeating the question.
    reprompt = prompt[len(prompt.rstrip()):]
    current_prompt = prompt
    while True:
        response = input(current_prompt).strip().lower()
        response = response[:1] if response else ""
        print("")
        if response in choices:
            return response
        print(hint)
        current_prompt = reprompt
