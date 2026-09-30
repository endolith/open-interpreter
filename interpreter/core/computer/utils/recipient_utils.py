def format_to_recipient(text, recipient):
    return f"@@@RECIPIENT:{recipient}@@@CONTENT:{text}@@@END"


def parse_for_recipient(content):
    if content.startswith("@@@RECIPIENT:") and "@@@END" in content:
        parts = content.split("@@@")
        recipient = parts[1].split(":")[1]
        # Split on the first colon only: the content itself may contain colons
        # (a URL, a timestamp, a Windows path, a dict repr). Splitting without
        # maxsplit=1 silently truncated everything after the content's own first
        # colon, so "http://example.com" came back as "http".
        new_content = parts[2].split(":", 1)[1]
        return recipient, new_content
    return None, content
