"""Streaming corruption regression tests for MessageBlock and CodeBlock."""

import io
import re
from typing import Callable, Iterable, Optional

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

LONG_POWERSHELL_LINE = (
    "Get-Service wuauserv | Set-Service -StartupType Disabled"
)


def assert_all_chars_present(
    source: str,
    output: str,
    *,
    ignore: Optional[Callable[[str], str]] = None,
) -> None:
    """Every non-whitespace character in source must appear in output in order."""

    def norm(s: str) -> str:
        s = re.sub(r"[*`#]", "", s)
        s = re.sub(r"(\d+)\.", r"\1", s)
        s = re.sub(r"\s+", "", s)
        return ignore(s) if ignore else s

    src = norm(source)
    out = norm(output)
    if not src:
        return
    i = 0
    for ch in src:
        j = out.find(ch, i)
        assert j != -1, f"Missing character {ch!r} after position {i} in output"
        i = j + 1


def assert_message_tokens_present(source: str, output: str) -> None:
    """Assert rendered output contains the semantic content from a markdown message."""
    textified = textify_markdown_code_blocks(source)
    plain = output.replace(" ", "")
    for line in textified.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        token = re.sub(r"[*`#]", "", stripped)
        token = re.sub(r"^\d+\.\s*", "", token)
        compact = re.sub(r"\s+", "", token)
        if len(compact) >= 4:
            assert compact in plain, f"Missing {token!r} in rendered output"


def stream_chunks(content: str, chunk_size: int = 3) -> Iterable[str]:
    for i in range(0, len(content), chunk_size):
        yield content[i : i + chunk_size]


def make_console(width: int, height: int) -> Console:
    return Console(
        file=io.StringIO(),
        width=width,
        height=height,
        force_terminal=True,
        legacy_windows=False,
        emoji=False,
    )


def bind_block_console(block, console: Console) -> None:
    """Attach a test console to a block that already started Live on a default console."""
    stop_live_display(block.live)
    block.live = Live(
        console=console,
        auto_refresh=False,
        vertical_overflow="ellipsis",
        redirect_stdout=False,
        redirect_stderr=False,
    )
    block.live.start()


def capture_message_block_stream(
    content: str,
    *,
    width: int = 80,
    height: int = 24,
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


def capture_two_message_turns(
    first: str,
    second: str,
    *,
    width: int = 80,
    height: int = 24,
) -> tuple[str, str]:
    console = make_console(width, height)

    block1 = MessageBlock()
    bind_block_console(block1, console)
    for chunk in stream_chunks(first, chunk_size=5):
        block1.add_content(chunk)
    block1.finalize()
    block1.end()
    after_first = console.file.getvalue()

    block2 = MessageBlock()
    bind_block_console(block2, console)
    for chunk in stream_chunks(second, chunk_size=5):
        block2.add_content(chunk)
    block2.finalize()
    block2.end()

    return after_first, console.file.getvalue()


def capture_code_then_message(
    code: str,
    message: str,
    *,
    width: int = 80,
    height: int = 24,
) -> str:
    console = make_console(width, height)

    code_block = CodeBlock(interpreter=None)
    bind_block_console(code_block, console)
    code_block.language = "python"
    code_block.active_line = 1
    for chunk in stream_chunks(code, chunk_size=2):
        code_block.code += chunk
        code_block.refresh(cursor=True)
    code_block.end()

    msg_block = MessageBlock()
    bind_block_console(msg_block, console)
    for chunk in stream_chunks(message, chunk_size=4):
        msg_block.add_content(chunk)
    msg_block.finalize()
    msg_block.end()

    return console.file.getvalue()


TERMINAL_SIZES = [
    (80, 24),
    (100, 30),
    (120, 40),
    (60, 20),
]


@pytest.mark.parametrize("width,height", TERMINAL_SIZES)
def test_list_with_indented_fence_preserves_content(width, height):
    output = capture_message_block_stream(
        LIST_WITH_INDENTED_FENCE, width=width, height=height
    )
    assert_message_tokens_present(LIST_WITH_INDENTED_FENCE, output)
    for n in range(1, 8):
        assert str(n) in output
    assert "StartupType" in output.replace(" ", "")


@pytest.mark.parametrize("width,height", TERMINAL_SIZES)
def test_long_powershell_line_chars_not_lost(width, height):
    content = f"4. Item:\n   ```powershell\n   {LONG_POWERSHELL_LINE}\n   ```\n"
    output = capture_message_block_stream(content, width=width, height=height)
    assert_all_chars_present(LONG_POWERSHELL_LINE, output)


def test_multi_turn_first_message_unchanged():
    first = LIST_WITH_INDENTED_FENCE
    second = "Follow-up paragraph after the list."
    after_first, after_second = capture_two_message_turns(first, second)
    assert_message_tokens_present(first, after_first)
    assert_message_tokens_present(first, after_second)
    assert "Follow-up" in after_second


def test_code_block_hello_world_survives_later_message():
    output = capture_code_then_message(
        'print("hello world")',
        "Second assistant message with more text.",
        width=80,
        height=24,
    )
    assert 'print("hello world")' in output
    assert "python" in output.lower()
    assert "Second assistant" in output


def test_resize_mid_stream():
    console = make_console(120, 40)
    block = MessageBlock()
    bind_block_console(block, console)
    half = len(LIST_WITH_INDENTED_FENCE) // 2
    for chunk in stream_chunks(LIST_WITH_INDENTED_FENCE[:half], chunk_size=4):
        block.add_content(chunk)
    console.width = 80
    console._width = 80
    block._last_width = 120
    for chunk in stream_chunks(LIST_WITH_INDENTED_FENCE[half:], chunk_size=4):
        block.add_content(chunk)
    block.finalize()
    block.end()
    output = console.file.getvalue()
    assert_message_tokens_present(LIST_WITH_INDENTED_FENCE, output)


def test_stop_live_display_does_not_double_refresh():
    """stop_live_display must not call refresh() before stop()."""
    console = make_console(80, 24)
    live = create_live_display(console)
    live.start()
    live.update("line one\nline two\nline three\n", refresh=True)
    stop_live_display(live)
    output = console.file.getvalue()
    cursor_ups = output.count("\x1b[A")
    assert cursor_ups <= 6



def test_thinking_stream_preserves_content_above():
    """Live refreshes during Thinking streaming must not erase committed output above."""
    marker = "PERMANENT_MARKER_ABOVE_THINKING"
    thinking = (
        "The user is saying that in the original seal, the quill emoji was the "
        "problem, not the bee. Let me look at that seal again and compare."
    )
    console = make_console(100, 30)
    console.print(marker)
    block = MessageBlock()
    bind_block_console(block, console)
    block.reasoning_mode = True
    for chunk in stream_chunks(thinking, chunk_size=1):
        block.add_content(chunk)
    assert marker in console.file.getvalue()
    block.finalize()
    block.end()
    output = console.file.getvalue()
    assert marker in output
    assert "quill emoji" in output


def test_thinking_stream_does_not_over_erase_lines():
    """Many Thinking refreshes must not inflate erase height and wipe scrollback."""
    marker = "LINE_ABOVE_12345"
    console = make_console(80, 24)
    console.print(marker)
    block = MessageBlock()
    bind_block_console(block, console)
    block.reasoning_mode = True
    payload = "Word " * 80
    for chunk in stream_chunks(payload, chunk_size=1):
        block.add_content(chunk)
    output_mid = console.file.getvalue()
    assert marker in output_mid
    block.finalize()
    block.end()
    output = console.file.getvalue()
    assert marker in output
    # Over-erasing often leaves long blank runs where marker content was eaten
    marker_idx = output.find(marker)
    assert marker_idx != -1
    after_marker = output[marker_idx + len(marker) : marker_idx + len(marker) + 200]
    assert after_marker.count("\n") < 30 or "Word" in after_marker

def test_code_panel_intact_through_finalize():
    """Code panel title and borders survive finalize (confirmation path)."""
    console = make_console(80, 24)
    code_block = CodeBlock(interpreter=None)
    bind_block_console(code_block, console)
    code_block.language = "python"
    code_block.code = 'print("Hello, world!")'
    code_block.active_line = 1
    code_block.refresh(cursor=False)
    code_block.finalize()
    output = console.file.getvalue()
    assert 'print("Hello, world!")' in output
    assert "python" in output.lower()

def test_stop_live_display_clamps_tall_shape():
    """stop_live_display must clamp _shape down, never inflate it (which eats scrollback)."""
    console = make_console(80, 24)
    console.print("KEEP_ME")
    live = create_live_display(console)
    live.start()
    live.update("streaming\n" * 3, refresh=True)
    live._live_render._shape = (80, 40)
    stop_live_display(live, max_erase_rows=8)
    output = console.file.getvalue()
    assert "KEEP_ME" in output
    cursor_ups = output.count("\x1b[A")
    assert cursor_ups <= 12
