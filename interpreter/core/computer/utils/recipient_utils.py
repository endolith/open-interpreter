def format_to_recipient(text, recipient):
    return f"@@@RECIPIENT:{recipient}@@@CONTENT:{text}@@@END"


_RECIPIENT_PREFIX = "@@@RECIPIENT:"
_CONTENT_SEPARATOR = "@@@CONTENT:"
_END_MARKER = "@@@END"


def parse_for_recipient(content):
    """Split a recipient-tagged message into (recipient, payload).

    Parse the fixed prefix, separator, and final end marker instead of
    splitting on every "@@@": the payload is user text and may contain the
    marker itself, which the old split truncated at (#376). The first
    "@@@CONTENT:" separates the header, so a colon inside the payload (a URL,
    timestamp, or Windows path) is left intact.
    """
    if not content.startswith(_RECIPIENT_PREFIX):
        return None, content
    if not content.endswith(_END_MARKER):
        return None, content
    end = len(content) - len(_END_MARKER)
    header = content[len(_RECIPIENT_PREFIX) : end]
    recipient, separator, payload = header.partition(_CONTENT_SEPARATOR)
    if not separator:
        return None, content
    return recipient, payload
