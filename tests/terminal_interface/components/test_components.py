from unittest import mock

from interpreter.terminal_interface.components.code_block import CodeBlock
from interpreter.terminal_interface.components.message_block import (
    MessageBlock,
    textify_markdown_code_blocks,
)


def test_textify_markdown_code_blocks_rewrites_fence_language():
    text = "Intro\n```python\nprint(1)\n```\nDone"
    result = textify_markdown_code_blocks(text)
    assert "```text" in result
    assert "```python" not in result


def test_textify_leaves_non_code_unchanged():
    text = "No code here"
    assert textify_markdown_code_blocks(text) == text


def test_message_block_refresh_updates_live():
    block = MessageBlock()
    block.message = "Hello **world**"
    with mock.patch.object(block, "live") as live:
        block.refresh(cursor=False)
        live.update.assert_called_once()
        live.refresh.assert_called_once()


def test_code_block_end_clears_active_line():
    block = CodeBlock()
    block.active_line = 3
    block.code = "x = 1"
    with mock.patch.object(block, "refresh"):
        with mock.patch.object(block.live, "stop"):
            block.end()
    assert block.active_line is None
