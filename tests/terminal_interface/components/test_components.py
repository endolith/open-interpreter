from io import StringIO
from types import SimpleNamespace
from unittest import mock

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from interpreter.terminal_interface.components.code_block import CodeBlock
from interpreter.terminal_interface.components.message_block import (
    MessageBlock,
    textify_markdown_code_blocks,
)


def _render(renderable):
    """Render a rich object to plain text so tests can assert on its contents."""
    buffer = StringIO()
    Console(file=buffer, color_system=None, width=80).print(renderable)
    return buffer.getvalue()


def test_textify_markdown_code_blocks_rewrites_fence_language():
    """Code fences are rewritten to use the text language tag for display."""
    text = "Intro\n```python\nprint(1)\n```\nDone"
    result = textify_markdown_code_blocks(text)
    assert "```text" in result
    assert "```python" not in result


def test_textify_leaves_non_code_unchanged():
    """Plain text without code fences passes through unchanged."""
    text = "No code here"
    assert textify_markdown_code_blocks(text) == text


def test_message_block_refresh_updates_live():
    """MessageBlock.refresh updates and refreshes the Rich live display."""
    block = MessageBlock()
    block.message = "Hello **world**"
    with mock.patch.object(block, "live") as live:
        block.refresh(cursor=False)
        live.update.assert_called_once()
        live.refresh.assert_called_once()


def test_code_block_end_clears_active_line():
    """CodeBlock.end clears active_line and stops the live display."""
    block = CodeBlock()
    block.active_line = 3
    block.code = "x = 1"
    with mock.patch.object(block, "refresh"):
        with mock.patch.object(block.live, "stop"):
            block.end()
    assert block.active_line is None


def test_code_block_active_line_is_highlighted():
    """CodeBlock.refresh styles the active line row with a white background."""
    block = CodeBlock()
    block.code = "a = 1\nb = 2"
    block.active_line = 1
    captured = {}
    block.live.update = lambda group: captured.update(group=group)
    block.live.refresh = lambda: None

    block.refresh(cursor=True)

    table = captured["group"].renderables[1].renderable
    assert isinstance(table, Table)
    assert [row.style for row in table.rows] == ["black on white", None]


def test_code_block_cursor_follows_highlight_setting():
    """CodeBlock.refresh appends the cursor bullet only when active-line
    highlighting is enabled (the interpreter opt-out is honored)."""
    with_cursor = CodeBlock()
    with_cursor.code = "x = 1"
    captured_with = {}
    with_cursor.live.update = lambda group: captured_with.update(group=group)
    with_cursor.live.refresh = lambda: None
    with_cursor.refresh(cursor=True)
    assert "\u25cf" in _render(captured_with["group"])

    without = CodeBlock(interpreter=SimpleNamespace(highlight_active_line=False))
    without.code = "x = 1"
    captured_without = {}
    without.live.update = lambda group: captured_without.update(group=group)
    without.live.refresh = lambda: None
    without.refresh(cursor=True)
    assert "\u25cf" not in _render(captured_without["group"])


def test_code_block_skips_refresh_without_content():
    """CodeBlock.refresh is a no-op when there is no code and no output."""
    block = CodeBlock()
    block.live.update = mock.Mock()
    block.live.refresh = mock.Mock()

    block.refresh(cursor=True)

    block.live.update.assert_not_called()
    block.live.refresh.assert_not_called()


def test_code_block_shows_output_panel_only_when_present():
    """CodeBlock.refresh renders an output panel when output is set, and leaves
    it blank (a bare string) otherwise."""
    with_output = CodeBlock()
    with_output.code = "x = 1"
    with_output.output = "hello"
    captured_with = {}
    with_output.live.update = lambda group: captured_with.update(group=group)
    with_output.live.refresh = lambda: None
    with_output.refresh(cursor=False)
    assert isinstance(captured_with["group"].renderables[2], Panel)

    without_output = CodeBlock()
    without_output.code = "x = 1"
    captured_without = {}
    without_output.live.update = lambda group: captured_without.update(group=group)
    without_output.live.refresh = lambda: None
    without_output.refresh(cursor=False)
    assert captured_without["group"].renderables[2] == ""


def test_code_block_stores_interpreter_highlight_flag():
    """CodeBlock reads the interpreter's highlight_active_line at construction."""
    block = CodeBlock(interpreter=SimpleNamespace(highlight_active_line=False))
    assert block.highlight_active_line is False
    assert CodeBlock().highlight_active_line is None


def test_textify_toggles_fence_languages():
    """textify_markdown_code_blocks rewrites the opening fence of every code
    block to ```text but leaves the closing ``` fences alone."""
    text = "```python\nx\n```\n```json\ny\n```"
    result = textify_markdown_code_blocks(text)
    assert result == "```text\nx\n```\n```text\ny\n```"


def test_textify_handles_indented_fences():
    """textify_markdown_code_blocks matches fences even with leading whitespace."""
    result = textify_markdown_code_blocks("  ```python\nx\n  ```")
    assert result.startswith("```text")


def test_message_block_cursor_bullet():
    """MessageBlock.refresh appends the cursor bullet when cursor is enabled."""
    block = MessageBlock()
    block.message = "Hello **world**"
    captured = {}
    block.live.update = lambda panel: captured.update(panel=panel)
    block.live.refresh = lambda: None

    block.refresh(cursor=True)
    assert "\u25cf" in _render(captured["panel"])

    captured.clear()
    block.refresh(cursor=False)
    assert "\u25cf" not in _render(captured["panel"])


def test_block_types_are_distinct():
    """MessageBlock and CodeBlock identify themselves with distinct types."""
    assert MessageBlock().type == "message"
    assert CodeBlock().type == "code"
