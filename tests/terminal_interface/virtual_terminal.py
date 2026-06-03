"""Terminal emulator for Rich Live / ANSI regression tests.

Pytest sets TERM=dumb, so Rich Live skips cursor motion on StringIO.  Force
xterm-256color and feed output through pyte so tests can detect:

- Large blank gaps in scrollback (Live over-erase)
- Thinking panel preview vs final vertical position shift
- Committed markdown blocks erased by inflated LiveRender._shape
"""

from __future__ import annotations

import io
import os
import re
from typing import Iterable, List

import pyte
from pyte import HistoryScreen, Stream
from rich.console import Console
from rich.live import Live

_FORCED_TERM = "xterm-256color"
os.environ["TERM"] = _FORCED_TERM


class VirtualTerminal:
    """pyte-backed terminal with scrollback history."""

    def __init__(self, width: int = 80, height: int = 24, history: int = 5000) -> None:
        self.width = width
        self.height = height
        self.screen = HistoryScreen(width, height, history=history)
        self.stream = Stream(self.screen)

    def feed(self, data: str) -> None:
        self.stream.feed(data)

    def _row_str(self, row) -> str:
        if isinstance(row, str):
            return row
        if isinstance(row, dict):
            chars = [" "] * self.width
            for x, char in row.items():
                chars[x] = char.data
            return "".join(chars).rstrip()
        return "".join(row)

    def all_rows(self) -> List[str]:
        return [self._row_str(r) for r in self.screen.history.top] + [
            self._row_str(r) for r in self.screen.display
        ]

    def max_blank_run(self) -> int:
        max_run = 0
        run = 0
        for row in self.all_rows():
            if not self._row_str(row).strip():
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return max_run

    def max_blank_run_in_history(self) -> int:
        max_run = 0
        run = 0
        for row in self.screen.history.top:
            if not self._row_str(row).strip():
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return max_run

    def line_indices(self, needle: str) -> List[int]:
        return [i for i, row in enumerate(self.all_rows()) if needle in row]

    def count_panel_titles(self, title: str = "Thinking") -> int:
        return sum(1 for row in self.all_rows() if title in row)


    def count_panel_titles_on_display(self, title: str = "Thinking") -> int:
        return sum(1 for row in self.screen.display if title in self._row_str(row))

    def max_blank_run_on_display(self) -> int:
        max_run = 0
        run = 0
        for row in self.screen.display:
            if not self._row_str(row).strip():
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return max_run

    def plain_text(self) -> str:
        return "\n".join(self.all_rows())


class TTYStringIO(io.StringIO):
    """StringIO that Rich treats as an interactive terminal."""

    def __init__(self, terminal: VirtualTerminal) -> None:
        super().__init__()
        self.terminal = terminal
        self.raw = ""

    def isatty(self) -> bool:
        return True

    def flush(self) -> None:
        pass

    def write(self, s: str) -> int:
        self.raw += s
        self.terminal.feed(s)
        return len(s)


def _patch_terminal_size(width: int, height: int) -> os.terminal_size:
    """Align os.get_terminal_size with the emulated console dimensions."""
    size = os.terminal_size((width, height))
    import interpreter.terminal_interface.components.message_block as message_block
    import interpreter.terminal_interface.utils.streaming_markdown as streaming_markdown

    streaming_markdown.os.get_terminal_size = lambda *a, **k: size
    message_block.os.get_terminal_size = lambda *a, **k: size
    return size


def make_tty_console(
    width: int = 80, height: int = 24
) -> tuple[Console, VirtualTerminal, TTYStringIO]:
    _patch_terminal_size(width, height)
    vt = VirtualTerminal(width=width, height=height)
    buf = TTYStringIO(vt)
    console = Console(
        file=buf,
        width=width,
        height=height,
        force_terminal=True,
        force_interactive=True,
        legacy_windows=False,
        emoji=False,
        _environ={**os.environ, "TERM": _FORCED_TERM},
    )
    vt._raw_sink = buf
    return console, vt, buf


def make_dumb_console(
    width: int = 100, height: int = 30
) -> tuple[Console, io.StringIO]:
    """StringIO console for content-presence tests (no ANSI simulation)."""
    buf = io.StringIO()
    console = Console(
        file=buf,
        width=width,
        height=height,
        force_terminal=True,
        legacy_windows=False,
        emoji=False,
        _environ={**os.environ, "TERM": "dumb"},
    )
    return console, buf


def bind_block_console(block, console: Console) -> None:
    if block.live.is_started:
        block.live.stop()
    block.live = Live(
        console=console,
        auto_refresh=False,
        vertical_overflow="ellipsis",
        redirect_stdout=False,
        redirect_stderr=False,
    )
    block.live.start()


def stream_chunks(content: str, chunk_size: int = 3) -> Iterable[str]:
    for i in range(0, len(content), chunk_size):
        yield content[i : i + chunk_size]


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


def count_erase_sequences(text: str) -> int:
    return text.count("\x1b[2K")


def panel_top_line(vt: VirtualTerminal, title: str = "Thinking") -> int | None:
    for i, row in enumerate(vt.all_rows()):
        if title in row:
            return i
    return None


def max_blank_run_in_raw(raw: str) -> int:
    """Count consecutive empty lines in ANSI-stripped terminal output."""
    max_run = 0
    run = 0
    for line in strip_ansi(raw).splitlines():
        if not line.strip():
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def assert_blank_run_lte(vt: VirtualTerminal, max_run: int, *, raw: str | None = None) -> None:
    """Fail on large blank gaps in captured terminal output.

    pyte history counts erase cycles as blank rows; raw ANSI output reflects
    what actually scrolls into terminal scrollback on ConPTY/xterm.
    """
    if raw is None and getattr(vt, "_raw_sink", None) is not None:
        raw = vt._raw_sink.raw
    actual = max_blank_run_in_raw(raw) if raw is not None else vt.max_blank_run_in_history()
    assert actual <= max_run, (
        f"Terminal output contains {actual} consecutive blank lines (max {max_run})"
    )


def assert_history_blank_run_lte(vt: VirtualTerminal, max_run: int) -> None:
    actual = vt.max_blank_run_in_history()
    assert actual <= max_run, (
        f"Scrollback contains {actual} consecutive blank lines (max {max_run})"
    )


def assert_panel_position_stable(
    vt: VirtualTerminal,
    title: str,
    before_line: int | None,
    *,
    tolerance: int = 0,
) -> None:
    assert before_line is not None, f"Expected {title!r} panel before finalize"
    after_line = panel_top_line(vt, title)
    assert after_line is not None, f"Expected {title!r} panel after finalize"
    delta = abs(after_line - before_line)
    assert delta <= tolerance, (
        f"{title!r} panel moved from line {before_line} to {after_line} "
        f"(delta {delta}, max {tolerance})"
    )


def assert_single_panel(vt: VirtualTerminal, title: str = "Thinking") -> None:
    """One panel body on screen (pyte may keep ghost borders from Live erase)."""
    bodies = set()
    for row in vt.screen.display:
        s = vt._row_str(row)
        if "│" not in s:
            continue
        inner = s.split("│", 2)[1].strip() if "│" in s else ""
        inner = inner.rstrip("●").strip()
        if len(inner) > 5:
            bodies.add(inner)
    assert len(bodies) <= 1, (
        f"Expected one {title!r} panel body on screen, found {len(bodies)}"
    )


def assert_anchor_preserved(vt: VirtualTerminal, anchor: str) -> None:
    assert anchor in vt.plain_text(), f"Lost anchor {anchor!r}"


def assert_content_preserved(vt: VirtualTerminal, *tokens: str) -> None:
    text = vt.plain_text()
    for token in tokens:
        assert token in text, f"Missing {token!r} in rendered output"


def get_tty_raw(buf: TTYStringIO) -> str:
    return buf.raw
