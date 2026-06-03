import os
import re
import shutil

from rich.box import MINIMAL, ROUNDED
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich.padding import Padding

from .base_block import BaseBlock
from ..utils.display_constants import PADDING_MESSAGE, PADDING_PANEL
from ..utils.streaming_markdown import (
    detect_complete_block,
    calculate_window_size,
    create_sliding_window_display,
    create_live_display,
    stop_live_display,
    textify_markdown_code_blocks,
)


class MessageBlock(BaseBlock):
    def __init__(self):
        super().__init__()

        # Override the Live display with our streaming configuration
        self.live = create_live_display(self.live.console)  # Use our streaming Live display
        self.live.start()

        self.type = "message"
        self.message = ""
        self.buffer = ""
        self.completed_blocks = []
        self.viewport_fraction = 0.3  # Increase from 0.2 to 0.3 for better visibility
        self.debug = False  # Enable debug mode to show colored borders
        # When True, do not commit complete blocks to console; keep everything in buffer until replace_content.
        # Used for reasoning/thinking so the blockquote replace truly replaces the streamed raw text.
        self.reasoning_mode = False
        try:
            self._last_width = os.get_terminal_size().columns
        except:
            self._last_width = shutil.get_terminal_size().columns

    def _max_live_erase_rows(self, viewport_lines):
        """Upper bound on Live rows we may erase; clamping prevents eating scrollback above."""
        overhead = 4 if (self.reasoning_mode or self.debug) else 1
        return max(1, viewport_lines + overhead + 1)

    def refresh(self, cursor=True):
        """Process new content and render complete blocks incrementally."""
        # Force re-detection of terminal size to handle window narrowing during streamed output
        self.live.console._width = None
        self.live.console._height = None

        try:
            current_size = os.get_terminal_size()
        except:
            current_size = shutil.get_terminal_size()

        current_width = current_size.columns
        if current_width != self._last_width:
            # If the terminal was resized during a live display, the terminal's
            # automatic reflow will have shifted our cursor position, leading to
            # "garbled" text as Live tries to clear an area that doesn't exist anymore.
            # We restart the live display to anchor it to a fresh position.
            viewport_lines = calculate_window_size(self.live.console, self.viewport_fraction)
            stop_live_display(
                self.live, max_erase_rows=self._max_live_erase_rows(max(viewport_lines, 1)))
            self._last_width = current_width
            self.live = create_live_display(self.live.console)
            self.live.start()

        # In reasoning mode, never commit blocks to console so replace_content can replace the whole buffer.
        if not self.reasoning_mode:
            block_result = detect_complete_block(self.buffer)
        else:
            block_result = None

        if block_result:
            block_text, next_line_begin = block_result

            # De-stylize any code blocks in markdown to differentiate from Code Blocks
            content = textify_markdown_code_blocks(block_text)

            # Render the complete block directly to console (above the Live viewport).
            markdown = Markdown(content.strip())

            was_started = self.live.is_started
            if was_started:
                viewport_lines = calculate_window_size(self.live.console, self.viewport_fraction)
                stop_live_display(
                    self.live,
                    max_erase_rows=self._max_live_erase_rows(max(viewport_lines, 1)),
                )

            if self.debug:
                panel = Panel(markdown, box=ROUNDED, border_style="green")
                self.live.console.print(panel)
            else:
                padded_markdown = Padding(markdown, PADDING_MESSAGE)
                self.live.console.print(padded_markdown)

            if was_started:
                self.live = create_live_display(self.live.console)
                self.live.start()

            self.completed_blocks.append(content)

            lines = self.buffer.split('\n')
            remaining_lines = lines[next_line_begin:]
            self.buffer = '\n'.join(remaining_lines)

        if self.buffer.strip():
            viewport_lines = calculate_window_size(self.live.console, self.viewport_fraction)

            if viewport_lines < 1:
                viewport_lines = 3

            width_offset = 6 if (self.reasoning_mode or self.debug) else 4

            formatted_buffer = create_sliding_window_display(
                self.live.console, self.buffer.split('\n'), viewport_lines, self.debug,
                base_style="cyan" if self.reasoning_mode else None, width_offset=width_offset)

            if cursor and isinstance(formatted_buffer, Text):
                formatted_buffer += "●"
            elif cursor and isinstance(formatted_buffer, Group):
                formatted_buffer.renderables[-1] += "●"

            if self.reasoning_mode:
                streaming_panel = Panel(formatted_buffer, box=ROUNDED, border_style="cyan", title="Thinking")
                padded_buffer = Padding(streaming_panel, PADDING_PANEL)
                self.live.update(padded_buffer, refresh=True)
            elif self.debug:
                streaming_panel = Panel(formatted_buffer, box=ROUNDED, border_style="blue")
                self.live.update(streaming_panel, refresh=True)
            else:
                padded_buffer = Padding(formatted_buffer, PADDING_MESSAGE)
                self.live.update(padded_buffer, refresh=True)
        else:
            self.live.update("", refresh=True)

    def add_content(self, content):
        """Add new content to the buffer and process it."""
        self.buffer += content
        self.refresh(cursor=True)

    def replace_content(self, content):
        """Replace the entire buffer (e.g. when streamed raw reasoning is replaced with blockquote-formatted version)."""
        self.buffer = content
        self.refresh(cursor=False)

    def finalize(self):
        """Render any remaining content when the message is complete."""
        viewport_lines = calculate_window_size(self.live.console, self.viewport_fraction)
        stop_live_display(
            self.live, max_erase_rows=self._max_live_erase_rows(max(viewport_lines, 1)))

        if self.buffer.strip():
            try:
                content = textify_markdown_code_blocks(self.buffer)
                if self.reasoning_mode:
                    markdown = Markdown(content.strip(), style="cyan")
                    panel = Panel(markdown, box=ROUNDED, border_style="cyan", title="Thinking")
                    padded_markdown = Padding(panel, PADDING_PANEL)
                    self.live.console.print(padded_markdown)
                else:
                    markdown = Markdown(content.strip())
                    if self.debug:
                        panel = Panel(markdown, box=ROUNDED, border_style="red")
                        self.live.console.print(panel)
                    else:
                        padded_markdown = Padding(markdown, PADDING_MESSAGE)
                        self.live.console.print(padded_markdown)
            except (IndexError, ValueError, TypeError):
                self.live.console.print(self.buffer)

        self.buffer = ""
