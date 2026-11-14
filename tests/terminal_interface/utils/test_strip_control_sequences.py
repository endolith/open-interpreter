import pytest

from interpreter.terminal_interface.utils.strip_control_sequences import (
    strip_control_sequences,
)


def test_strips_csi_mouse_reporting_sequences():
    """A stored mouse-enable sequence must not reach the terminal (#169)."""
    payload = "before \x1b[?1000h\x1b[?1003h\x1b[?1006h after"
    assert strip_control_sequences(payload) == "before  after"
    assert "\x1b" not in strip_control_sequences(payload)


@pytest.mark.parametrize(
    "payload, expected",
    [
        ("title\x1b]0;poison\x07rest", "titlerest"),  # OSC terminated by BEL
        ("title\x1b]0;poison\x1b\\rest", "titlerest"),  # OSC terminated by ST
        ("colour\x1b[31mred\x1b[0m", "colourred"),  # SGR colour changes
        ("cursor\x1b[2J\x1b[H", "cursor"),  # clear screen and home cursor
        ("keys\x1b[=5u", "keys"),  # modify key reporting
    ],
)
def test_strips_terminal_sequences_of_each_family(payload, expected):
    """CSI, OSC and other ESC sequences are removed as whole units."""
    assert strip_control_sequences(payload) == expected


def test_strips_raw_c0_del_and_c1_controls_but_keeps_layout():
    """Control bytes outside the escape syntax are removed; newline and tab stay."""
    payload = "a\x00b\x07c\r\x7fd\x9be\nf\tg"
    assert strip_control_sequences(payload) == "abcde\nf\tg"


def test_unterminated_escape_introducer_is_removed():
    """A truncated sequence leaves its visible parameters but loses the ESC byte."""
    assert strip_control_sequences("x\x1b[?1000") == "x[?1000"


def test_keeps_printable_unicode_and_plain_text():
    """Ordinary text, including non-ASCII, passes through byte-for-byte."""
    plain = "hello world"
    assert strip_control_sequences(plain) == plain
    assert strip_control_sequences("héllo 🙂 世界") == "héllo 🙂 世界"


def test_non_string_values_pass_through():
    """The helper is a no-op for values that are not strings."""
    assert strip_control_sequences(None) is None
