#!/usr/bin/env python3
"""
Example script demonstrating how to print complex markdown-formatted text
using the rich library with streaming simulation. This includes code blocks,
tables, nested elements, and various markdown features.
"""

import time
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from markdown_it import MarkdownIt


def parse_markdown_into_blocks(markdown_text):
    """
    Parse markdown into blocks using markdown-it-py's token stream.
    Extract complete blocks by processing tokens in order and grouping related tokens.
    """
    md = MarkdownIt().enable("strikethrough").enable("table")
    tokens = md.parse(markdown_text)

    lines = markdown_text.split('\n')
    blocks = []

    # Process tokens sequentially to build complete blocks
    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Process block-level tokens (both opening and self-closing)
        if token.block and token.map and (token.nesting == 1 or token.nesting == 0):
            line_begin, line_end = token.map

            # Extract the complete block from original source
            block_lines = lines[line_begin:line_end]
            block_text = '\n'.join(block_lines)

            if block_text.strip():
                # Return the original block text along with token type and tag
                blocks.append((block_text, token.type, token.tag))

            # Skip to the end of this block to avoid processing nested content
            if token.nesting == 1:  # Only skip for opening tokens
                nesting_level = 1
                i += 1
                while i < len(tokens) and nesting_level > 0:
                    if tokens[i].nesting == 1:
                        nesting_level += 1
                    elif tokens[i].nesting == -1:
                        nesting_level -= 1
                    i += 1
                continue

        i += 1

    return blocks


def stream_markdown_blocks(console, markdown_text, chunk_size=10, delay=0.1, window_fraction=0.4):
    """
    Stream markdown text block by block using sliding window approach.
    Each block streams with a sliding window, then renders the complete block.

    Args:
        console: Rich Console instance
        markdown_text: The markdown text to stream
        chunk_size: Number of characters per chunk
        delay: Delay between chunks in seconds
        window_fraction: Fraction of terminal height to use for sliding window (default: 0.4 = 40%)
    """
    blocks = parse_markdown_into_blocks(markdown_text)

    # Calculate window size based on terminal height
    terminal_height = console.size.height
    window_lines = max(8, int(terminal_height * window_fraction))  # Minimum 8 lines

    prev_element_new_line = False

    for block_text, token_type, token_tag in blocks:
        # Insert a blank line BEFORE this block if the previous element requested a newline
        # Matches Rich's renderer behavior where most elements set new_line=True
        if prev_element_new_line:
            console.print()

        # Create a console with highlighting disabled for streaming
        plain_console = Console(highlight=False)

        # Create a new Live object for each block
        with Live(console=plain_console, refresh_per_second=20,
                  vertical_overflow="ellipsis") as live:

            # Stream the current block with sliding window
            block_accumulated = ""
            for i in range(0, len(block_text), chunk_size):
                chunk = block_text[i:i + chunk_size]
                block_accumulated += chunk

                # Split into lines and keep only the last window_lines
                current_lines = block_accumulated.split('\n')
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

            # After streaming is complete, show the full rendered block
            time.sleep(0.2)  # Brief pause before final display
            try:
                live.update(Markdown(block_text))
            except (IndexError, ValueError, TypeError):
                # If markdown parsing fails, show plain text
                live.update(block_text)

        # Decide if a newline should be inserted BEFORE the next element
        # In Rich, HorizontalRule sets new_line=False, everything else True
        prev_element_new_line = (token_type != "hr")


def main():
    console = Console()

    # Load markdown content from external file
    with open('sample_markdown.md', 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # Stream the markdown text block by block with sliding window
    stream_markdown_blocks(console, markdown_text, chunk_size=10, delay=0.1, window_fraction=0.1)

if __name__ == "__main__":
    main()
