def prompt_choice(prompt, choices):
    """
    Prompt until the user enters one of the given single-character choices.
    Returns the choice. choices e.g. ('y', 'n') or ('y', 'a', 'n').
    """
    choices = tuple(c.lower() for c in choices)
    hint = "Please enter " + ", ".join(f"'{c}'" for c in choices) + ".\n"
    while True:
        response = input(prompt).strip().lower()
        response = response[:1] if response else ""
        print("")
        if response in choices:
            return response
        print(hint)
