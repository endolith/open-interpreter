from rich.box import MINIMAL
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .base_block import BaseBlock


class CodeBlock(BaseBlock):
    """
    Code Blocks display code and outputs in different languages. You can also set the active_line!
    """

    def __init__(self, interpreter=None):
        super().__init__()

        self.type = "code"
        self.highlight_active_line = (
            interpreter.highlight_active_line if interpreter else None
        )

        # Define these for IDE auto-completion
        self.language = ""
        self.output = ""
        self.code = ""
        self.code_printed = False
        self.active_line = None
        self.margin_top = True

    def end(self):
        self.active_line = None
        self.refresh(cursor=False)
        super().end()

    def print_code_before_run(self):
        """Print the code panel permanently before the command is dispatched.

        The run block normally first renders when the first execution chunk
        arrives. A command that writes straight to the terminal (sudo's
        password prompt, for instance) can therefore print before the panel
        is drawn, and the panel then repaints over its line. Printing the
        panel here, and starting a fresh live display for the output, keeps
        it above everything the command itself prints.
        """
        if not self.code or self.code_printed:
            return

        console = self.live.console
        self.live.stop()
        console.print(self._render_group(cursor=False))
        # A new live display starts below the printed panel, so later refreshes
        # can only repaint the execution output, never the panel or the
        # command's own terminal output above it.
        self.live = Live(
            auto_refresh=False, console=console, vertical_overflow="visible"
        )
        self.live.start()
        # The panel is permanent now, so the live region shows only the output.
        # Keep the source around: the active-line handler looks up the current
        # action through this block's code.
        self.code_printed = True

    def refresh(self, cursor=True):
        if not self.output and (not self.code or self.code_printed):
            return

        self.live.update(self._render_group(cursor=cursor))
        self.live.refresh()

    def _render_group(self, cursor=True):
        """Build the Rich renderable for the current code and output."""
        # Get code
        code = self.code

        # Create a table for the code
        code_table = Table(
            show_header=False, show_footer=False, box=None, padding=0, expand=True
        )
        code_table.add_column()

        # Add cursor only if active line highliting is true
        if cursor and (
            self.highlight_active_line
            if self.highlight_active_line is not None
            else True
        ):
            code += "●"

        # Add each line of code to the table
        code_lines = code.strip().split("\n")
        for i, line in enumerate(code_lines, start=1):
            if i == self.active_line and (
                self.highlight_active_line
                if self.highlight_active_line is not None
                else True
            ):
                # This is the active line, print it with a white background
                syntax = Syntax(
                    line, self.language, theme="bw", line_numbers=False, word_wrap=True
                )
                code_table.add_row(syntax, style="black on white")
            else:
                # This is not the active line, print it normally
                syntax = Syntax(
                    line,
                    self.language,
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True,
                )
                code_table.add_row(syntax)

        # Create a panel for the code
        code_panel = Panel(code_table, box=MINIMAL, style="on #272722")

        # Create a panel for the output (if there is any)
        if self.output == "" or self.output == "None":
            output_panel = ""
        else:
            output_panel = Panel(self.output, box=MINIMAL, style="#FFFFFF on #3b3b37")

        # Create a group with the code table and output panel. Once the panel
        # has been printed permanently, only the output belongs in the live
        # region; rendering the code again would duplicate it.
        group_items = [] if self.code_printed else [code_panel]
        group_items.append(output_panel)
        if self.margin_top:
            # This adds some space at the top. Just looks good!
            group_items = [""] + group_items
        return Group(*group_items)
