from interpreter.core.utils.sanitize_terminal_input import (
    sanitize_terminal_input,
    strip_terminal_state_sequences,
)


def test_sgr_mouse_reports_stripped_from_input():
    """SGR mouse motion/press/release reports (byobu/tmux mouse mode) vanish."""
    raw = "hello\x1b[<35;25;57M\x1b[<35;26;57M world\x1b[<0;80;53m"
    assert sanitize_terminal_input(raw) == "hello world"


def test_terminal_query_replies_stripped_from_input():
    """Cursor-position (R) and window-size (t) replies plus stray ST vanish."""
    raw = "tmux 3.4\x1b\\\x1b[61;4R\x1b[61;5R\x1b[1;1R\x1b[4;427;624t"
    assert sanitize_terminal_input(raw) == "tmux 3.4"


def test_full_reported_garbage_line_cleans_to_prefix_text():
    """The exact shape from the bug report leaves only the typed prefix."""
    raw = ":>|tmux 3.4\x1b\\\x1b[61;4R\x1b[<35;25;57M\x1b[<0;80;53m"
    assert sanitize_terminal_input(raw) == ":>|tmux 3.4"


def test_mouse_only_input_cleans_to_empty():
    """A prompt containing nothing but mouse motion is treated as no input."""
    raw = "\x1b[<35;25;57M\x1b[<35;26;57M\x1b[<0;80;53m"
    assert sanitize_terminal_input(raw) == ""


def test_trailing_partial_sequence_dropped():
    """A report cut off at the end of the read (e.g. pasted `ESC[<35;`) goes."""
    assert sanitize_terminal_input("hi\x1b[<35;") == "hi"
    assert sanitize_terminal_input("hi\x1b[") == "hi"
    assert sanitize_terminal_input("hi\x1b") == "hi"


def test_ordinary_text_and_paste_untouched():
    """Normal typing, tabs, unicode, and pasted SGR-color docs survive."""
    text = "how many files are on my desktop?\tcafé ✓"
    assert sanitize_terminal_input(text) == text
    # Interior color-code-looking text the user pasted is kept (only a
    # *trailing* fragment is dropped); Alt-key ESC chords are kept too.
    assert sanitize_terminal_input("a\x1bb") == "a\x1bb"


def test_legacy_and_focus_reports_stripped():
    """X10 mouse (ESC[M + 3 bytes) and focus in/out (ESC[I/O) vanish."""
    assert sanitize_terminal_input("x\x1b[MABCy") == "xy"
    assert sanitize_terminal_input("a\x1b[Ib\x1b[Oc") == "abc"


def test_private_modes_stripped_from_output_colors_kept():
    """DECSET mouse-tracking/box-drawing modes go, but SGR colors stay."""
    raw = "\x1b[?1000h\x1b[?1006h\x1b[32mgreen\x1b[0m\x1b[?2004l"
    assert strip_terminal_state_sequences(raw) == "\x1b[32mgreen\x1b[0m"


def test_device_queries_stripped_from_output():
    """DSR/XTWINOPS/version/mode queries go — their replies would land in input."""
    raw = "probe\x1b[6n\x1b[18t\x1b[>0q\x1b[?1004$p\x1b[?u done"
    assert strip_terminal_state_sequences(raw) == "probe done"


def test_cursor_shape_and_colors_survive_output_strip():
    """Cursor-shape (space before q) and SGR colors are benign and kept."""
    raw = "\x1b[1 q\x1b[32mgreen\x1b[0m"
    assert strip_terminal_state_sequences(raw) == raw


def test_osc_hyperlink_stripped_from_output():
    """OSC window-title/hyperlink payloads never reach the saved log."""
    raw = "see \x1b]8;;http://example.com\x1b\\link\x1b]8;;\x1b\\ done"
    assert strip_terminal_state_sequences(raw) == "see link done"


def test_fast_path_no_esc_returns_same_object():
    """Inputs without ESC skip regex work and return the identical string."""
    text = "plain input"
    assert sanitize_terminal_input(text) is text
    assert strip_terminal_state_sequences(text) is text
