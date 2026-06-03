"""Incremental markdown rendering regression tests for MessageBlock and CodeBlock."""

import io
import re
from typing import Iterable, Optional

import pytest
from rich.console import Console
from rich.live import Live

from interpreter.terminal_interface.components.code_block import CodeBlock
from interpreter.terminal_interface.components.message_block import MessageBlock
from interpreter.terminal_interface.utils.streaming_markdown import (
    create_live_display,
    stop_live_display,
    textify_markdown_code_blocks,
)

NESTED_LIST = """1. Level 1
   - Level 2
     - Level 3
       - Level 4
         - Level 5
           - Level 6
             - Level 7
               - Level 8
                 - Level 9
"""

NESTED_LIST_WITH_HEADING = """## Level 2: Nested Lists Until You Die

1. Fruits
   - Tropical
     - Mango
       - Alphonso
         - Premium Alphonso
           - Sourced from: India
             - State: Maharashtra
               - Region: Ratnagiri
"""

MULTI_BLOCK_MESSAGE = """Intro paragraph before the list.

1. **First item**

2. **Second item**

---

1. Level 1
   - Level 2
     - Level 3
"""

LONG_POWERSHELL_LINE = (
    "Get-Service wuauserv | Set-Service -StartupType Disabled"
)

LIST_WITH_INDENTED_FENCE = """1. **First**

2. **Second**

3. **Third**

4. **Run PowerShell**:
   ```powershell
   Get-Service wuauserv | Set-Service -StartupType Disabled
   Get-Service bits | Set-Service -StartupType Disabled
   ```

5. **Fifth**

6. **Sixth**

7. **Seventh**
"""


def stream_chunks(content: str, chunk_size: int = 3) -> Iterable[str]:
    for i in range(0, len(content), chunk_size):
        yield content[i : i + chunk_size]


def make_console(width: int, height: int) -> Console:
    """Dumb StringIO console for content-presence tests (no ANSI/Live simulation)."""
    buf = io.StringIO()
    return Console(
        file=buf,
        width=width,
        height=height,
        force_terminal=True,
        legacy_windows=False,
        emoji=False,
        _environ={"TERM": "dumb"},
    )


def bind_block_console(block, console: Console) -> None:
    """Attach a test console; MessageBlock starts Live in __init__ on a default console."""
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


def assert_no_excessive_blank_runs(output: str, max_consecutive_newlines: int = 12) -> None:
    """Fail if output contains huge blank gaps (Live over-erase / spacing regression)."""
    for match in re.finditer(r"\n+", output):
        if len(match.group()) > max_consecutive_newlines:
            start = max(0, match.start() - 40)
            end = min(len(output), match.end() + 40)
            context = repr(output[start:end])
            raise AssertionError(
                f"Excessive blank run: {len(match.group())} consecutive newlines "
                f"(max {max_consecutive_newlines}). Context: {context}"
            )


def assert_message_tokens_present(source: str, output: str) -> None:
    """Assert rendered output contains semantic content from a markdown message."""
    textified = textify_markdown_code_blocks(source)
    plain = re.sub(r"\s+", "", output)
    for line in textified.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```") or stripped == "---":
            continue
        token = re.sub(r"[*`#]", "", stripped)
        token = re.sub(r"^\d+\.\s*", "", token)
        token = re.sub(r"^[-•]\s*", "", token)
        compact = re.sub(r"\s+", "", token)
        if len(compact) >= 4:
            assert compact in plain, f"Missing {token!r} in rendered output"


def capture_message_stream(
    content: str,
    *,
    width: int = 100,
    height: int = 30,
    chunk_size: int = 3,
    reasoning: bool = False,
) -> str:
    console = make_console(width, height)
    block = MessageBlock()
    bind_block_console(block, console)
    block.reasoning_mode = reasoning
    for chunk in stream_chunks(content, chunk_size):
        block.add_content(chunk)
    block.finalize()
    block.end()
    return console.file.getvalue()


def capture_two_turns(first: str, second: str, *, width: int = 100, height: int = 30) -> str:
    console = make_console(width, height)
    for content in (first, second):
        block = MessageBlock()
        bind_block_console(block, console)
        for chunk in stream_chunks(content, chunk_size=5):
            block.add_content(chunk)
        block.finalize()
        block.end()
    return console.file.getvalue()


TERMINAL_SIZES = [(80, 24), (100, 30), (120, 40)]


@pytest.mark.parametrize("width,height", TERMINAL_SIZES)
def test_nested_list_levels_preserved(width, height):
    output = capture_message_stream(NESTED_LIST, width=width, height=height)
    for level in range(1, 10):
        assert f"Level {level}" in output
    assert_no_excessive_blank_runs(output)


@pytest.mark.parametrize("width,height", TERMINAL_SIZES)
def test_nested_list_with_heading(width, height):
    output = capture_message_stream(
        NESTED_LIST_WITH_HEADING, width=width, height=height, chunk_size=2
    )
    assert "Nested Lists Until You Die" in output
    for token in ["Fruits", "Tropical", "Mango", "Alphonso", "Maharashtra", "Ratnagiri"]:
        assert token in output
    assert_no_excessive_blank_runs(output)


@pytest.mark.parametrize("width,height", TERMINAL_SIZES)
def test_multi_block_incremental_commit(width, height):
    output = capture_message_stream(
        MULTI_BLOCK_MESSAGE, width=width, height=height, chunk_size=2
    )
    assert "Intro paragraph" in output
    assert "First item" in output.replace("*", "")
    for level in range(1, 4):
        assert f"Level {level}" in output
    assert_no_excessive_blank_runs(output)


@pytest.mark.parametrize("width,height", TERMINAL_SIZES)
def test_list_with_indented_code_fence(width, height):
    output = capture_message_stream(
        LIST_WITH_INDENTED_FENCE, width=width, height=height
    )
    assert_message_tokens_present(LIST_WITH_INDENTED_FENCE, output)
    assert "StartupType" in output.replace(" ", "")
    for n in range(1, 8):
        assert str(n) in output
    assert_no_excessive_blank_runs(output)


@pytest.mark.parametrize("width,height", TERMINAL_SIZES)
def test_long_line_not_truncated(width, height):
    content = f"4. Item:\n   ```powershell\n   {LONG_POWERSHELL_LINE}\n   ```\n"
    output = capture_message_stream(content, width=width, height=height)
    assert "StartupType" in output.replace(" ", "")
    assert "Disabled" in output


def test_multi_turn_prior_content_preserved():
    first = LIST_WITH_INDENTED_FENCE
    second = "Follow-up after the nested list test."
    output = capture_two_turns(first, second)
    assert_message_tokens_present(first, output)
    assert "Follow-up" in output
    assert_no_excessive_blank_runs(output)


def test_reasoning_stream_and_finalize():
    thinking = (
        "The user wants a nested list again. I'll write plain markdown "
        "with careful indentation and no code fences around it."
    )
    console = make_console(100, 30)
    console.print("MARKER_ABOVE")
    block = MessageBlock()
    bind_block_console(block, console)
    block.reasoning_mode = True
    for chunk in stream_chunks(thinking, chunk_size=2):
        block.add_content(chunk)
    assert "MARKER_ABOVE" in console.file.getvalue()
    block.finalize()
    block.end()
    output = console.file.getvalue()
    assert "MARKER_ABOVE" in output
    assert "nested list" in output.lower()
    assert output.count("Thinking") >= 1
    assert_no_excessive_blank_runs(output)


def test_reasoning_then_message():
    console = make_console(100, 30)
    console.print("ANCHOR_LINE")

    thinking_block = MessageBlock()
    bind_block_console(thinking_block, console)
    thinking_block.reasoning_mode = True
    for chunk in stream_chunks("Thinking about nested lists.", chunk_size=3):
        thinking_block.add_content(chunk)
    thinking_block.finalize()
    thinking_block.end()

    msg_block = MessageBlock()
    bind_block_console(msg_block, console)
    for chunk in stream_chunks(NESTED_LIST, chunk_size=3):
        msg_block.add_content(chunk)
    msg_block.finalize()
    msg_block.end()

    output = console.file.getvalue()
    assert "ANCHOR_LINE" in output
    for level in range(1, 10):
        assert f"Level {level}" in output
    assert_no_excessive_blank_runs(output)


def test_code_block_panel_survives_later_message():
    console = make_console(80, 24)
    code_block = CodeBlock(interpreter=None)
    bind_block_console(code_block, console)
    code_block.language = "python"
    code_block.active_line = 1
    for chunk in stream_chunks('print("Hello, world!")', chunk_size=2):
        code_block.code += chunk
        code_block.refresh(cursor=True)
    code_block.end()

    msg_block = MessageBlock()
    bind_block_console(msg_block, console)
    for chunk in stream_chunks("Second turn message.", chunk_size=4):
        msg_block.add_content(chunk)
    msg_block.finalize()
    msg_block.end()

    output = console.file.getvalue()
    assert 'print("Hello, world!")' in output
    assert "python" in output.lower()
    assert_no_excessive_blank_runs(output)


def test_marker_survives_many_thinking_refreshes():
    console = make_console(80, 24)
    console.print("SCROLLBACK_MARKER")
    block = MessageBlock()
    bind_block_console(block, console)
    block.reasoning_mode = True
    payload = "Word " * 60
    for chunk in stream_chunks(payload, chunk_size=1):
        block.add_content(chunk)
    block.finalize()
    block.end()
    output = console.file.getvalue()
    assert "SCROLLBACK_MARKER" in output
    assert_no_excessive_blank_runs(output)


def test_char_by_char_nested_list():
    output = capture_message_stream(NESTED_LIST, chunk_size=1, width=100, height=30)
    for level in range(1, 10):
        assert f"Level {level}" in output
    assert_no_excessive_blank_runs(output)

def count_live_stops(content: str, *, width: int = 100, height: int = 30, chunk_size: int = 2) -> int:
    stops = 0
    original_stop = Live.stop

    def counting_stop(self):
        nonlocal stops
        if self.is_started:
            stops += 1
        return original_stop(self)

    console = make_console(width, height)
    block = MessageBlock()
    bind_block_console(block, console)

    import unittest.mock as mock

    with mock.patch.object(Live, "stop", counting_stop):
        for chunk in stream_chunks(content, chunk_size):
            block.add_content(chunk)
        block.finalize()
        block.end()
    return stops


def test_incremental_commit_stops_are_bounded():
    """Incremental commits stop Live per block; total stops must stay bounded."""
    stops = count_live_stops(MULTI_BLOCK_MESSAGE, chunk_size=2)
    assert 1 <= stops <= 5, f"Expected bounded Live stops, got {stops}"


def test_no_standalone_hr_during_incremental_commit():
    """Markdown --- separators must not render as a lone full-width rule mid-stream."""
    output = capture_message_stream(MULTI_BLOCK_MESSAGE, chunk_size=2)
    # Rich HR rules are long runs of box-drawing characters.
    hr_runs = [len(m.group()) for m in re.finditer(r"[─━—]{20,}", output)]
    assert not hr_runs or max(hr_runs) < 80, (
        f"Standalone HR rendered during streaming: longest run {max(hr_runs)} chars"
    )


def test_user_flow_heading_list_after_thinking():
    """Reproduce heading + nested list after a Thinking panel (reported corruption)."""
    console = make_console(120, 40)
    console.print("USER_PROMPT")

    thinking_block = MessageBlock()
    bind_block_console(thinking_block, console)
    thinking_block.reasoning_mode = True
    thinking = (
        "I'll write a pure nested list with careful indentation and no code fence."
    )
    for chunk in stream_chunks(thinking, chunk_size=2):
        thinking_block.add_content(chunk)
    thinking_block.finalize()
    thinking_block.end()

    msg_block = MessageBlock()
    bind_block_console(msg_block, console)
    for chunk in stream_chunks(NESTED_LIST_WITH_HEADING, chunk_size=2):
        msg_block.add_content(chunk)
    msg_block.finalize()
    msg_block.end()

    output = console.file.getvalue()
    assert "USER_PROMPT" in output
    assert "Nested Lists Until You Die" in output
    for token in ["Fruits", "Tropical", "Mango", "Ratnagiri"]:
        assert token in output
    assert_no_excessive_blank_runs(output, max_consecutive_newlines=6)



def test_stop_live_display_no_double_refresh():
    console = make_console(80, 24)
    live = create_live_display(console)
    live.start()
    live.update("line one\nline two\n", refresh=True)
    stop_live_display(live)
    cursor_ups = console.file.getvalue().count("\x1b[A")
    assert cursor_ups <= 6


# --- Virtual-terminal (ANSI) regression tests --------------------------------
# pyte simulates cursor-up/erase so StringIO dumb-terminal blind spots are covered.

from tests.terminal_interface.virtual_terminal import (
    assert_anchor_preserved,
    assert_blank_run_lte,
    assert_content_preserved,
    assert_panel_position_stable,
    assert_single_panel,
    bind_block_console,
    make_tty_console,
    panel_top_line,
    stream_chunks as vt_stream_chunks,
)
from interpreter.terminal_interface.components.message_block import MessageBlock


def _prior_lines(console, n: int = 5) -> None:
    for i in range(n):
        console.print(f"SCROLLBACK_{i}")


def test_thinking_panel_position_stable_on_finalize():
    """Thinking preview and finalized panel must occupy the same scrollback line."""
    console, vt, buf = make_tty_console(80, 12)
    _prior_lines(console, 2)
    block = MessageBlock()
    bind_block_console(block, console)
    block.reasoning_mode = True
    thinking = "I'll rewrite the nested list with careful indentation."
    for chunk in vt_stream_chunks(thinking, chunk_size=2):
        block.add_content(chunk)
    before_line = panel_top_line(vt)
    block.finalize()
    block.end()
    assert_panel_position_stable(vt, "Thinking", before_line)
    assert_single_panel(vt)
    assert_blank_run_lte(vt, max_run=4)
    assert_anchor_preserved(vt, "SCROLLBACK_0")


def test_thinking_replace_content_then_finalize_stable():
    """replace_content (blockquote) must finalize at the preview line, not below it."""
    console, vt, buf = make_tty_console(80, 12)
    console.print("USER_PROMPT")
    block = MessageBlock()
    bind_block_console(block, console)
    block.reasoning_mode = True
    raw = "I'll write plain markdown with careful indentation."
    for chunk in vt_stream_chunks(raw, chunk_size=2):
        block.add_content(chunk)
    formatted = "> I'll write plain markdown with careful indentation."
    block.replace_content(formatted)
    before_line = panel_top_line(vt)
    block.finalize()
    block.end()
    assert_panel_position_stable(vt, "Thinking", before_line)
    assert_single_panel(vt)
    assert_content_preserved(vt, "careful indentation")
    assert_anchor_preserved(vt, "USER_PROMPT")


def test_multi_block_incremental_commit_no_history_blank_gaps():
    """Incremental commits on a short terminal must not leave blank scrollback runs."""
    console, vt, buf = make_tty_console(80, 12)
    _prior_lines(console, 3)
    block = MessageBlock()
    bind_block_console(block, console)
    for chunk in vt_stream_chunks(MULTI_BLOCK_MESSAGE, chunk_size=2):
        block.add_content(chunk)
    block.finalize()
    block.end()
    assert_content_preserved(vt, "Intro paragraph", "Level 3")
    assert_blank_run_lte(vt, max_run=4)


def test_nested_list_with_heading_no_history_blank_gaps():
    console, vt, buf = make_tty_console(80, 12)
    _prior_lines(console, 2)
    block = MessageBlock()
    bind_block_console(block, console)
    for chunk in vt_stream_chunks(NESTED_LIST_WITH_HEADING, chunk_size=2):
        block.add_content(chunk)
    block.finalize()
    block.end()
    assert_content_preserved(vt, "Nested Lists Until You Die", "Ratnagiri")
    assert_blank_run_lte(vt, max_run=4)


def test_thinking_then_message_no_scrollback_corruption():
    """Full user-reported flow: Thinking panel, then nested list with heading."""
    console, vt, buf = make_tty_console(100, 16)
    console.print("USER_PROMPT")
    tb = MessageBlock()
    bind_block_console(tb, console)
    tb.reasoning_mode = True
    for chunk in vt_stream_chunks(
        "I'll write a pure nested list with careful indentation.", chunk_size=2
    ):
        tb.add_content(chunk)
    think_before = panel_top_line(vt)
    tb.finalize()
    tb.end()

    mb = MessageBlock()
    bind_block_console(mb, console)
    for chunk in vt_stream_chunks(NESTED_LIST_WITH_HEADING, chunk_size=2):
        mb.add_content(chunk)
    mb.finalize()
    mb.end()

    assert_panel_position_stable(vt, "Thinking", think_before)
    assert_single_panel(vt)
    assert_anchor_preserved(vt, "USER_PROMPT")
    assert_content_preserved(vt, "Ratnagiri")
    assert_blank_run_lte(vt, max_run=4)


def test_char_by_char_nested_list_no_history_blank_gaps():
    console, vt, buf = make_tty_console(80, 12)
    _prior_lines(console, 4)
    block = MessageBlock()
    bind_block_console(block, console)
    for chunk in vt_stream_chunks(NESTED_LIST, chunk_size=1):
        block.add_content(chunk)
    block.finalize()
    block.end()
    for level in range(1, 10):
        assert_content_preserved(vt, f"Level {level}")
    assert_blank_run_lte(vt, max_run=4)


def test_committed_heading_not_erased_by_finalize():
    """Heading committed incrementally must survive list streaming and finalize."""
    console, vt, buf = make_tty_console(80, 12)
    block = MessageBlock()
    bind_block_console(block, console)
    for chunk in vt_stream_chunks(NESTED_LIST_WITH_HEADING, chunk_size=2):
        block.add_content(chunk)
    block.finalize()
    block.end()
    assert_content_preserved(vt, "Nested Lists Until You Die", "Ratnagiri")
