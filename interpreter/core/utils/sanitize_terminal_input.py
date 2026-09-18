"""Strip unsolicited terminal reports from interactive input and saved output.

Terminals never send mouse events on their own: byobu/tmux mouse mode keeps
events in the multiplexer, and the app only receives mouse reports after
something arms app-side tracking with DECSET (e.g. ``ESC[?1006h``). The known
route is a poisoned conversation log — a full-screen TUI program's output
(cursor addressing, device queries like ``ESC[6n``, tracking enables like
``ESC[?1000h``) captured verbatim into a saved console message, then
re-emitted raw when history replays on resume, arming that pane. From then on
every mouse movement arrives on stdin as SGR reports (``ESC[<35;25;57M``) and
each embedded query byte solicits replies (``ESC[61;4R``, ``ESC[4;427;624t``)
that land in the next ``input()`` call — the "mouse moving garbage" at the
``> `` prompt, polluting readline history and the log further.

Separately, program output can contain terminal *state-changing* sequences
(DECSET ``ESC[?1000h`` to enable mouse tracking, ``ESC[?2004h`` for
bracketed paste, OSC hyperlinks, ...). If those are saved verbatim into the
conversation ``.json`` log, replaying the history on the next session
re-emits them and re-arms mouse tracking — so the garbage keeps coming back.
"""

import re

# SGR mouse reports (xterm 1006): ESC [ < Cb ; Cx ; Cy M/m.
# Covers press (M), release (m), motion (32/35...), wheel (64...).
_SGR_MOUSE_RE = re.compile(r"\x1b\[<[0-9;]*[mM]")

# Legacy X10/xterm mouse: ESC [ M followed by 3 raw bytes (button, x, y).
_LEGACY_MOUSE_RE = re.compile(r"\x1b\[M.{3}", re.DOTALL)

# Cursor-position / device-status replies: ESC [ ... R (e.g. ESC[61;4R).
_CPR_RE = re.compile(r"\x1b\[[0-9;]*R")

# XTWINOPS replies: ESC [ ... t (e.g. ESC[4;427;624t window-size report).
_XTWINOPS_RE = re.compile(r"\x1b\[[0-9;]*t")

# Focus in/out reports: ESC [ I / ESC [ O.
_FOCUS_RE = re.compile(r"\x1b\[[IO]")

# Lone String Terminator (ESC \) — the tail of an OSC/DCS reply whose head
# was consumed elsewhere (seen as `^[` in the reported garbage).
_ST_RE = re.compile(r"\x1b\\")

# A truncated report at the very end of the read (the terminal / paste buffer
# cut the line mid-sequence, e.g. a final `ESC[<35;`). The next input() call
# will deliver the remainder, which the complete-patterns above then catch —
# so dropping the fragment loses nothing the user typed.
_TRAILING_PARTIAL_RE = re.compile(r"\x1b(\[<?[0-9;]*)?$")


def sanitize_terminal_input(text: str) -> str:
    """Remove mouse/report escape sequences leaked into interactive input.

    Only strips byte patterns that can never be intentionally typed at the
    ``> `` prompt — mouse reports, terminal query replies, focus events.
    Regular text (including bracketed-paste bodies, tabs, unicode) passes
    through untouched, as do bare ESC uses like Alt-key chords.
    """
    if not text or "\x1b" not in text:
        return text
    cleaned = _SGR_MOUSE_RE.sub("", text)
    cleaned = _LEGACY_MOUSE_RE.sub("", cleaned)
    cleaned = _CPR_RE.sub("", cleaned)
    cleaned = _XTWINOPS_RE.sub("", cleaned)
    cleaned = _FOCUS_RE.sub("", cleaned)
    cleaned = _ST_RE.sub("", cleaned)
    # Only a trailing fragment can be partial; an interior ESC[ is either a
    # complete sequence handled above or (e.g. pasted code showing ESC[31m)
    # something the user actually pasted, which we must keep.
    cleaned = _TRAILING_PARTIAL_RE.sub("", cleaned)
    return cleaned


# DECSET/DECRST private-mode set/reset: ESC [ ? ... final-byte.
# This is the family that changes terminal *state*: mouse tracking
# (1000/1002/1003/1005/1006/1015/1016), bracketed paste (2004), alternate
# screen (1049), cursor visibility (25), etc. Plain SGR colors (ESC[31m —
# no `?`) are deliberately NOT matched so program colors survive.
_PRIVATE_MODE_RE = re.compile(r"\x1b\[[?][0-9;]*[@-~]")

# OSC sequences: ESC ] ... terminated by BEL or ST (window titles,
# hyperlinks). Re-emitting them on history replay renames the window or
# worse, so they never belong in the saved log.
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


# Device queries: DSR (ESC[6n), XTWINOPS (ESC[18t, and any ...t — replies
# included), XTVERSION (ESC[>0q), DECRQM (ESC[?...$p), kitty flags (ESC[?u).
# Re-emitting them solicits replies that land in the next input() call, so
# they must never persist in saved output. Plain cursor-shape (ESC[1 q, with
# a space) and SGR colors (final byte m) are deliberately not matched.
_QUERY_RE = re.compile(r"\x1b\[(?:[0-9;]*[nt]|[?>][0-9;]*q|\?[0-9;]*\$p|\?u)")


def strip_terminal_state_sequences(text: str) -> str:
    """Remove state-changing ANSI from program output before save/display.

    Keeps ordinary SGR colors (``ESC[31m``) intact — only sequences that flip
    terminal modes, carry OSC payloads, or solicit terminal replies are
    stripped, since those would otherwise persist into the conversation
    ``.json`` log and re-arm e.g. mouse tracking (or inject query replies
    into ``input()``) when the history is replayed next session.
    """
    if not text or "\x1b" not in text:
        return text
    cleaned = _SGR_MOUSE_RE.sub("", text)
    cleaned = _PRIVATE_MODE_RE.sub("", cleaned)
    cleaned = _QUERY_RE.sub("", cleaned)
    cleaned = _OSC_RE.sub("", cleaned)
    return cleaned
