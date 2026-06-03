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
            stop_live_display(self.live)
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
            # Do not stop/restart Live here — that erase cycle corrupts scrollback
            # above the Live anchor when blocks commit during streaming.
            markdown = Markdown(content.strip())

            if self.debug:
                # In debug mode, still use panel for visual distinction
                panel = Panel(markdown, box=ROUNDED, border_style="green")
                self.live.console.print(panel)
            else:
                # Print markdown directly with horizontal padding only (2 chars left/right)
                padded_markdown = Padding(markdown, PADDING_MESSAGE)
                self.live.console.print(padded_markdown)

            # Store the completed block
            self.completed_blocks.append(content)

            # Remove the rendered block from buffer using line numbers
            lines = self.buffer.split('\n')
            remaining_lines = lines[next_line_begin:]
            self.buffer = '\n'.join(remaining_lines)

            # If we removed content, refresh the viewport with remaining content
            if remaining_lines:
                # Continue to the streaming section below
                pass

        # Stream the remaining buffer content in the Live viewport
        if self.buffer.strip():
            # Calculate viewport size
            viewport_lines = calculate_window_size(self.live.console, self.viewport_fraction)

            # Ensure we have a reasonable viewport size
            if viewport_lines < 1:
                viewport_lines = 3  # Minimum viewport size

            # Calculate width offset for wrapping (padding/borders)
            # Default 4 (PADDING_MESSAGE is 2 left, 2 right)
            # reasoning_mode/debug use a Panel (+2 for borders) and PADDING_PANEL (4)
            width_offset = 6 if (self.reasoning_mode or self.debug) else 4

            # Create sliding window display for the buffer
            formatted_buffer = create_sliding_window_display(
                self.live.console, self.buffer.split('\n'), viewport_lines, self.debug,
                base_style="cyan" if self.reasoning_mode else None, width_offset=width_offset)

            # Add cursor if requested
            if cursor and isinstance(formatted_buffer, Text):
                formatted_buffer += "●"
            elif cursor and isinstance(formatted_buffer, Group):
                # If it's a Group with ellipsis, add cursor to the text part
                formatted_buffer.renderables[-1] += "●"

            # Wrap streaming content in a panel to match rendered content indentation.
            # refresh=True is required because create_live_display sets auto_refresh=False
            # (to prevent the background refresh thread from erasing sudo prompts printed
            # by child processes). Without auto_refresh, update() alone only stores the
            # renderable without rendering it; refresh=True forces an immediate render.
            if self.reasoning_mode:
                # Distinct style so thinking is visually separate from normal blockquotes
                streaming_panel = Panel(formatted_buffer, box=ROUNDED, border_style="cyan", title="Thinking")
                padded_buffer = Padding(streaming_panel, PADDING_PANEL)
                self.live.update(padded_buffer, refresh=True)
            elif self.debug:
                streaming_panel = Panel(formatted_buffer, box=ROUNDED, border_style="blue")
                self.live.update(streaming_panel, refresh=True)
            else:
                # Print streaming content directly with horizontal padding only (2 chars left/right)
                padded_buffer = Padding(formatted_buffer, PADDING_MESSAGE)
                self.live.update(padded_buffer, refresh=True)
        else:
            # Clear the live display if no buffer content
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
        stop_live_display(self.live)

        # Render any remaining buffer content as markdown
        if self.buffer.strip():
            try:
                # De-stylize any code blocks in markdown
                content = textify_markdown_code_blocks(self.buffer)
                if self.reasoning_mode:
                    # Text same color as box (cyan), markdown formatting preserved
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
                # Fallback to plain text if markdown parsing fails
                self.live.console.print(self.buffer)

        # Ensure no further streaming occurs during end()'s refresh
        self.buffer = ""
