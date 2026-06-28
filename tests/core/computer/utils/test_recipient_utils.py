from interpreter.core.computer.utils.recipient_utils import (
    format_to_recipient,
    parse_for_recipient,
)


def test_format_and_parse_round_trip():
    text = "Hello, user!"
    recipient = "user"
    formatted = format_to_recipient(text, recipient)
    parsed_recipient, parsed_content = parse_for_recipient(formatted)
    assert parsed_recipient == recipient
    assert parsed_content == text


def test_parse_plain_text_without_markers():
    content = "Just a normal message"
    recipient, parsed = parse_for_recipient(content)
    assert recipient is None
    assert parsed == content


def test_format_preserves_newlines():
    text = "Line1\nLine2 without colons"
    formatted = format_to_recipient(text, "assistant")
    _, parsed = parse_for_recipient(formatted)
    assert parsed == text
