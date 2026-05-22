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

        # Override the base Live display with our specialized streaming configuration.
        # Do NOT start the live display here — start it lazily in refresh() so that
        # creating a CodeBlock in plain_text_display mode (where refresh() is never
        # called) doesn't launch a background Rich display that intercepts print()
        # calls and scatters streaming tokens to separate lines.
        self.live = create_live_display(self.live.console)

        self.type = "code"
        self.highlight_active_line = (
            interpreter.highlight_active_line if interpreter else None
        )

        # Define these for IDE auto-completion
        self.language = ""
        self.target_path = ""  # edit tool: shown outside the code panel, not in self.code
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
        if self.live.is_started:
            self.live.update("")
            self.live.refresh()
            self.live.stop()

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
            if os.name == "nt" and syntax_language.lower() in ["cmd", "bash", "shell"]:
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

        if format_type == "code" and self.target_path:
            group_items.append(Text(self.target_path, style="dim"))

        group_items.append(panel)
        
        was_started = self.live.is_started
        if was_started:
            self.live.update("")
            self.live.refresh()
            self.live.stop()
            
        self.live.console.print(Padding(Group(*group_items), PADDING_PANEL))
        
        if was_started:
            from ..utils.streaming_markdown import create_live_display
            self.live = create_live_display(self.live.console)
            self.live.start()

    def refresh(self, cursor=True):
        """Process content, pop completed blocks, and update the Live sliding window."""
        # Lazy start: the live display is not started in __init__ so that creating a
        # CodeBlock in plain_text_display mode is side-effect-free.  Rich's start()
        # is idempotent (safe to call again after stop), so this is always safe here.
        if not self.live.is_started:
            self.live.start()

        # Ensure we have fresh terminal dimensions
        self.live.console._width = None
        self.live.console._height = None

        try:
            current_size = os.get_terminal_size()
        except:
            current_size = shutil.get_terminal_size()

        current_width = current_size.columns
        if current_width != self._last_width:
            # Re-start live display on resize (critical for Windows terminal reflow)
            if self.live.is_started:
                self.live.stop()
            self._last_width = current_width
            self.live = create_live_display(self.live.console)
            self.live.start()

        should_highlight = self.highlight_active_line if self.highlight_active_line is not None else True

        # If active line highlighting is disabled, we permanently render the code block when it finishes
        # streaming to avoid repeated refreshes corrupting terminal history.
        # We know it's finished streaming if self.active_line is not None OR self.output has content.
        if not should_highlight and self.code.strip() and (self.active_line is not None or self.output.strip()):
            self._print_permanent_block(self.code, "code")
            self.code_lines_popped += len(self.code.split('\n'))
            self.code = ""

        # Execution stdout/stderr is plain text, not markdown. detect_complete_block()
        # would split it into many top-level "paragraph" panels; keep one buffer and
        # print it once in finalize().

        # Render the remaining content
        if not self.code.strip() and not self.output.strip():
            self.live.update("")
            return

        # If highlighting is enabled AND execution has started (active_line is not None)
        if should_highlight and self.active_line is not None and self.code.strip():
            # Get code
            code = self.code

            # Create a table for the code
            code_table = Table(
                show_header=False, show_footer=False, box=None, padding=0, expand=True
            )
            code_table.add_column()

            # Add cursor only if active line highliting is true
            if cursor:
                code += "●"

            syntax_language = self.language
            if os.name == "nt" and syntax_language.lower() in ["cmd", "bash", "shell"]:
                syntax_language = "bat"

            # Add each line of code to the table
            code_lines = code.strip().split("\n")
            for i, line in enumerate(code_lines, start=1):
                if i == self.active_line:
                    # This is the active line, print it with a white background
                    syntax = Syntax(
                        line, syntax_language, theme="bw", line_numbers=False, word_wrap=True
                    )
                    code_table.add_row(syntax, style="black on white")
                else:
                    # This is not the active line, print it normally
                    syntax = Syntax(
                        line,
                        syntax_language,
                        theme="monokai",
                        line_numbers=False,
                        word_wrap=True,
                    )
                    code_table.add_row(syntax)

            # Create a panel for the code
            code_panel = Panel(code_table, box=MINIMAL, style="on #272722")

            group_items = []
            if self.target_path:
                group_items.append(Text(self.target_path, style="dim"))
            group_items.append(code_panel)

            # Create a panel for the output (if there is any).
            # We format remaining output as sliding window
            if self.output.strip():
                viewport_lines = calculate_window_size(self.live.console, self.viewport_fraction)
                viewport_lines = max(viewport_lines, 3)
                output_lines = self.output.strip().split('\n')
                
                formatted_output = create_sliding_window_display(
                    self.live.console, output_lines, viewport_lines, width_offset=6)
                
                # Escape so Rich does not interpret [brackets] as markup
                output_panel = Panel(formatted_output, box=MINIMAL, style="#FFFFFF on #3b3b37")
                group_items.append(output_panel)
                
            if self.margin_top:
                # This adds some space at the top. Just looks good!
                group_items = [""] + group_items
                self.margin_top = False
                
            # Create a group with the code table and output panel
            group = Group(*group_items)
            padded = Padding(group, PADDING_PANEL)

            # Update the live display
            self.live.update(padded)
            self.live.refresh()
            return

        # Otherwise, stream as raw non-highlighted text in a Live sliding window
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
        
        group_items = []
        if self.margin_top:
            group_items.append("")
            self.margin_top = False

        if self.target_path:
            group_items.append(Text(self.target_path, style="dim"))
            
        group_items.append(streaming_panel)
        # Update the live display
        self.live.update(Padding(Group(*group_items), PADDING_PANEL))
        self.live.refresh()

