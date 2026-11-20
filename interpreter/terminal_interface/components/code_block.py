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
        # Once the panel is permanent the live region may be showing the
        # single-line execution highlight. Repaint it even when the run
        # produced no output, so the stopped display does not keep the
        # highlight on screen.
        if self.code_printed:
            self.live.update(self._render_group(cursor=False))
            self.live.refresh()
        else:
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
        if (
            not self.output
            and self._active_code_line() is None
            and (not self.code or self.code_printed)
        ):
            return

        self.live.update(self._render_group(cursor=cursor))
        self.live.refresh()

    def _highlight_enabled(self):
        """Whether the interpreter asked for the active line to be highlighted."""
        return (
            self.highlight_active_line
            if self.highlight_active_line is not None
            else True
        )

    def _active_code_line(self):
        """The source line to highlight, or None when there is nothing to show.

        After the panel is printed permanently the live region no longer draws
        the whole block, but execution still needs to show which line is
        running. Returning the line here keeps that highlight available
        without rendering the code a second time.
        """
        if not self.code_printed or self.active_line is None:
            return None
        if not self._highlight_enabled():
            return None
        code_lines = self.code.strip().split("\n")
        if not isinstance(self.active_line, int) or not (
            1 <= self.active_line <= len(code_lines)
        ):
            return None
        return code_lines[self.active_line - 1]

    def _render_active_line(self):
        """Panel holding only the active line, for the printed-panel state."""
        line = self._active_code_line()
        if line is None:
            return None

        line_table = Table(
            show_header=False, show_footer=False, box=None, padding=0, expand=True
        )
        line_table.add_column()
        line_table.add_row(
            Syntax(line, self.language, theme="bw", line_numbers=False, word_wrap=True),
            style="black on white",
        )
        return Panel(line_table, box=MINIMAL, style="on #272722")

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
        # has been printed permanently, the code is already on screen, so the
        # live region draws only the active line (when highlighting is on) and
        # the output; rendering the code again would duplicate it.
        if self.code_printed:
            active_line_panel = self._render_active_line()
            group_items = [active_line_panel] if active_line_panel else []
        else:
            group_items = [code_panel]
        group_items.append(output_panel)
        if self.margin_top:
            # This adds some space at the top. Just looks good!
            group_items = [""] + group_items
        return Group(*group_items)
