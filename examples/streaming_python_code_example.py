#!/usr/bin/env python3
"""
Example script demonstrating how to stream a large Python code block (2 pages worth)
using the rich library. This shows a comprehensive Python class with various
programming concepts and patterns.
"""

import time
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.syntax import Syntax
from rich.text import Text
from rich.align import Align


def stream_python_code_with_live(console, code_text, chunk_size=15, delay=0.05, window_fraction=0.4):
    """
    Stream Python code with a sliding window of the last N lines, then show full result.

    Args:
        console: Rich Console instance
        code_text: The Python code text to stream
        chunk_size: Number of characters per chunk (default: 15)
        delay: Delay between chunks in seconds (default: 0.05)
        window_fraction: Fraction of terminal height to use for sliding window (default: 0.4 = 40%)
    """
    accumulated_text = ""
    lines = []

    # Extract the actual Python code (remove the markdown code block markers)
    code_content = code_text.replace("```python\n", "").replace("```", "")

    # Calculate window size based on terminal height
    terminal_height = console.size.height
    window_lines = max(8, int(terminal_height * window_fraction))  # Minimum 8 lines

    # Create a console with highlighting disabled for streaming
    plain_console = Console(highlight=False)

    with Live(console=plain_console, refresh_per_second=20,
              vertical_overflow="ellipsis") as live:
        for i in range(0, len(code_content), chunk_size):
            chunk = code_content[i:i + chunk_size]
            accumulated_text += chunk

            # Split into lines and keep only the last window_lines
            current_lines = accumulated_text.split('\n')
            if len(current_lines) > window_lines:
                display_lines = current_lines[-window_lines:]
                # Create a single Text object with centered red ellipsis
                display_text = Text()
                # Add centered ellipsis by padding it
                terminal_width = console.size.width
                ellipsis_padding = (terminal_width - 3) // 2  # Center the 3-character "..."
                display_text.append(" " * ellipsis_padding + "...", style="red")
                display_text.append("\n")
                display_text.append('\n'.join(display_lines))
            else:
                display_text = Text('\n'.join(current_lines))

            # Update with styled text
            live.update(display_text)
            time.sleep(delay)

        # After streaming is complete, show the full syntax-highlighted version
        time.sleep(0.5)  # Brief pause before final display
        syntax = Syntax(code_content, "python", theme="ansi_dark", background_color="default")
        live.update(syntax)


def main():
    console = Console()

    # Load Python code content from external file
    with open('sample_python_code.py', 'r', encoding='utf-8') as f:
        python_code_content = f.read()

    # Wrap in markdown code block for consistency
    python_code = f"```python\n{python_code_content}\n```"

    # Stream the Python code with sliding window, then show full result
    stream_python_code_with_live(console, python_code, chunk_size=15, delay=0.1, window_fraction=0.75)

if __name__ == "__main__":
    main()
