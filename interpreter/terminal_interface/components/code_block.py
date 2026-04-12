from rich.box import MINIMAL
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.padding import Padding
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
import shutil
import os

from .base_block import BaseBlock
from ..utils.display_constants import PADDING_PANEL
from ..utils.streaming_markdown import (
    detect_complete_block,
    calculate_window_size,
    create_sliding_window_display,
    create_live_display,
)


class CodeBlock(BaseBlock):
    """
    Code Blocks display code and outputs in different languages. You can also set the active_line!
    They now support incremental rendering to prevent terminal corruption with large outputs.
    """

    def __init__(self, interpreter=None):
        super().__init__()

        # Override the base Live display with our specialized streaming configuration
        self.live.stop()
        self.live = create_live_display(self.live.console)
        # We start it in refresh or let BaseBlock handle it? BaseBlock.init starts it.
        self.live.start()

        self.type = "code"
        self.highlight_active_line = (
            interpreter.highlight_active_line if interpreter else None
        )

        # Define these for IDE auto-completion
        self.language = ""
        self.output = ""
        self.code = ""
        self.active_line = None
        self.margin_top = True
        self.viewport_fraction = 0.3
        self.code_lines_popped = 0

        try:
            self._last_width = os.get_terminal_size().columns
        except:
            self._last_width = shutil.get_terminal_size().columns

    def end(self):
        self.active_line = None
        self.finalize()
        super().end()

    def finalize(self):
        """Render any remaining content permanently and clear the Live area."""
        self.live.update("")

        if self.code.strip():
            self._print_permanent_block(self.code, "code")
            # Update code_lines_popped to maintain relative line numbering
            self.code_lines_popped += len(self.code.split('\n'))
            self.code = ""

        if self.output.strip():
            self._print_permanent_block(self.output, "output")
            # We don't need to track output lines popped as active_line doesn't apply to them
            self.output = ""

    def _print_permanent_block(self, content, format_type):
        """Helper to render a completed segment and print it directly to console."""
        if not content.strip():
            return

        if format_type == "code":
            # Create the syntax-highlighted panel
            syntax_language = self.language
            if os.name == "nt" and syntax_language.lower() in ["shell", "bash"]:
                syntax_language = "bat"

            # We use a table to allow for potential line-specific styling if needed
            code_table = Table(
                show_header=False, show_footer=False, box=None, padding=0, expand=True
            )
            code_table.add_column()

            code_lines = content.strip().split("\n")
            for line in code_lines:
                syntax = Syntax(
                    line,
                    syntax_language,
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True,
                )
                code_table.add_row(syntax)

            panel = Panel(code_table, box=MINIMAL, style="on #272722")
        else:
            # Output panel
            panel = Panel(escape(content.strip()), box=MINIMAL, style="#FFFFFF on #3b3b37")

        group_items = []
        if self.margin_top:
            group_items.append("")
            self.margin_top = False

        group_items.append(panel)
        self.live.console.print(Padding(Group(*group_items), PADDING_PANEL))

    def refresh(self, cursor=True):
        """Process content, pop completed blocks, and update the Live sliding window."""
        # Ensure we have fresh terminal dimensions
        self.live.console._width = None
        self.live.console._height = None

        try:
            current_size = os.get_terminal_size()
        except:
            current_size = shutil.get_terminal_size()

        current_width = current_size.columns
        if current_width != self._last_width:
            self.live.stop()
            self._last_width = current_width
            self.live = create_live_display(self.live.console)
            self.live.start()

        # 1. Detect and pop complete blocks from code
        while True:
            block_result = detect_complete_block(self.code)
            if not block_result:
                break

            block_text, next_idx = block_result
            self._print_permanent_block(block_text, "code")

            lines = self.code.split('\n')
            self.code = '\n'.join(lines[next_idx:])
            self.code_lines_popped += next_idx

        # 2. Detect and pop complete blocks from output (if code is mostly finished/popped)
        if not self.code.strip():
            while True:
                block_result = detect_complete_block(self.output)
                if not block_result:
                    break

                block_text, next_idx = block_result
                self._print_permanent_block(block_text, "output")

                lines = self.output.split('\n')
                self.output = '\n'.join(lines[next_idx:])

        # 3. Render the remaining content in a Live sliding window
        if not self.code.strip() and not self.output.strip():
            self.live.update("")
            return

        # Prepare streaming buffer lines
        buffer_lines = []
        if self.code.strip():
            buffer_lines.extend(self.code.strip().split('\n'))
        if self.output.strip():
            if buffer_lines:
                buffer_lines.append("") # Spacer
            buffer_lines.extend(self.output.strip().split('\n'))

        # Calculate viewport size
        viewport_lines = calculate_window_size(self.live.console, self.viewport_fraction)
        viewport_lines = max(viewport_lines, 3)

        # Create sliding window display
        formatted_buffer = create_sliding_window_display(
            self.live.console, buffer_lines, viewport_lines, width_offset=6)

        # Add cursor if requested
        if cursor:
            if isinstance(formatted_buffer, Group):
                # If it's a Group (with ellipsis), add cursor to the last Text part
                last_renderable = formatted_buffer.renderables[-1]
                if isinstance(last_renderable, Text):
                    last_renderable.append("●")
                else:
                    formatted_buffer.renderables.append(Text("●"))
            elif isinstance(formatted_buffer, Text):
                formatted_buffer.append("●")

        # Wrap in a panel to match visual style
        title = f" {self.language} " if self.language else " Code "
        streaming_panel = Panel(formatted_buffer, box=MINIMAL, style="on #272722", title=title)
        self.live.update(Padding(streaming_panel, PADDING_PANEL))
        self.live.refresh()

