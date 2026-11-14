import re

# Everything a terminal may interpret as a command rather than as text: C0 and
# C1 control bytes, a lone DEL, and the ECMA-48 escape sequences that carry
# them. Newline and tab are deliberate text layout, so they are preserved.
#
# Order matters: the sequence patterns come first so a complete sequence is
# consumed as a unit, and the bare ESC fallback only removes a leftover
# introducer (e.g. an unterminated CSI) instead of leaving the terminal
# waiting for the rest of it.
_CONTROL_SEQUENCE = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC ... BEL or ST
    r"|\x1b\[[0-?]*[ -/]*[@-~]"  # CSI
    r"|\x1b[P^_X][^\x1b]*(?:\x1b\\)?"  # DCS/APC/PM/SOS ... ST
    r"|\x1b[@-Z\\-_]"  # other two-byte ESC sequences
    r"|\x1b"  # unterminated escape introducer
    r"|[\x00-\x08\x0b-\x1f\x7f-\x9f]"  # C0 (except \t, \n), DEL, C1
)


def strip_control_sequences(text):
    """Return ``text`` with terminal control sequences and raw control bytes removed.

    Conversation content is untrusted: tool output, pasted text and resumed
    history all flow into Rich renderers, which pass control bytes through to
    the terminal. A stored mouse-reporting sequence can therefore reconfigure
    the user's terminal every time the conversation is displayed. Strip the
    sequences at the display boundary so the stored message stays intact for
    the API and for exports.

    Newline and tab survive because they are ordinary text layout; everything
    else in the control ranges is removed, leaving the surrounding printable
    text in place.
    """
    if not isinstance(text, str):
        return text
    return _CONTROL_SEQUENCE.sub("", text)
