from interpreter.core.computer.utils.recipient_utils import (
    format_to_recipient,
    parse_for_recipient,
)


def test_format_and_parse_round_trip():
    """format_to_recipient and parse_for_recipient preserve recipient and content."""
    text = "Hello, user!"
    recipient = "user"
    formatted = format_to_recipient(text, recipient)
    parsed_recipient, parsed_content = parse_for_recipient(formatted)
    assert parsed_recipient == recipient
    assert parsed_content == text


def test_parse_plain_text_without_markers():
    """parse_for_recipient leaves plain text unchanged and returns no recipient."""
    content = "Just a normal message"
    recipient, parsed = parse_for_recipient(content)
    assert recipient is None
    assert parsed == content


def test_format_preserves_newlines():
    """format_to_recipient keeps embedded newlines through a round-trip parse."""
    text = "Line1\nLine2 without colons"
    formatted = format_to_recipient(text, "assistant")
    _, parsed = parse_for_recipient(formatted)
    assert parsed == text


def test_recipient_prefix_without_end_marker_is_not_parsed():
    """A string starting with the recipient prefix but lacking @@@END is plain text.

    The parse requires both the prefix and the end marker (`and`); with `or` a
    truncated marker would be treated as a recipient message and the content
    would be mangled instead of passed through.
    """
    content = "@@@RECIPIENT:user@@@CONTENT:hello"
    recipient, parsed = parse_for_recipient(content)
    assert recipient is None
    assert parsed == content


def test_end_marker_without_prefix_is_not_parsed():
    """A string containing @@@END but not starting with the prefix is plain text.

    The prefix check is what distinguishes a real recipient frame from ordinary
    text that happens to mention the end marker.
    """
    content = "some text @@@END here"
    recipient, parsed = parse_for_recipient(content)
    assert recipient is None
    assert parsed == content

def test_content_containing_a_colon_is_preserved():
    """Content with its own colon survives intact rather than being cut at that colon.

    The payload was parsed with an unbounded split, keeping only the field
    before the content's first colon, so a URL, timestamp, Windows path or dict
    repr was silently truncated ("http://example.com" came back as "http").
    Any chunk the terminal routes through parse_for_recipient is affected.

    The truncated-wrapper case is already covered by
    test_recipient_prefix_without_end_marker_is_not_parsed.
    """
    tagged = format_to_recipient("http://example.com", "assistant")
    recipient, content = parse_for_recipient(tagged)
    assert recipient == "assistant"
    assert content == "http://example.com"