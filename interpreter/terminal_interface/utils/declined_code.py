"""The message recorded when a user declines to run proposed code.

The terminal interface writes the refusal and the ``%jupyter`` exporter has to
recognize it again, so the text and the predicate that matches it live here
instead of being duplicated in both modules.
"""

DECLINED_CODE_MESSAGE = "User declined to run this code."


def is_declined_code_message(message):
    """Return True when ``message`` is the console record of a declined run.

    Refusals are stored as ``computer`` console output so they are displayed in
    the terminal but never replayed to the model as a user turn. No other
    console record carries this text, so the content identifies the refusal.
    """
    return (
        message.get("role") == "computer"
        and message.get("type") == "console"
        and message.get("content") == DECLINED_CODE_MESSAGE
    )
