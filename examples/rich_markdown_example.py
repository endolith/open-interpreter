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


def detect_complete_block(markdown_text):
    """
    Detect complete blocks by finding when the next block starts.
    Returns (complete_block_text, block_type) when a complete block is found.
    """
    try:
        md = MarkdownIt().enable("strikethrough").enable("table")
        tokens = md.parse(markdown_text)

        lines = markdown_text.split('\n')

        # Find block-level tokens (exclude inline tokens)
        block_tokens = []
        for token in tokens:
            if token.block and token.map and (token.nesting == 1 or token.nesting == 0) and token.type != 'inline':
                line_begin, line_end = token.map
                block_lines = lines[line_begin:line_end]
                block_text = '\n'.join(block_lines)
                if block_text.strip():
                    block_tokens.append((block_text, token.type, line_begin, line_end))

        # If we have at least 2 blocks, the first one is complete
        if len(block_tokens) >= 2:
            first_block_text, first_block_type, line_begin, line_end = block_tokens[0]
            return first_block_text, first_block_type, line_begin, line_end
        
        return None
    except Exception:
        return None


def break_into_words(text, chunk_size=3):
    """
    Break text into character-based chunks simulating LLM tokens.

    Args:
        text: The complete text to break up
        chunk_size: Number of characters per chunk (default: 3)

    Returns:
        List of character chunks simulating LLM tokens
    """
    words = []
    for i in range(0, len(text), chunk_size):
        words.append(text[i:i + chunk_size])
    return words


def create_display_text(buffer, window_lines, console):
    """Create display text with sliding window and ellipsis."""
    current_lines = buffer.split('\n')
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
        display_text = Text(buffer)
    return display_text


def process_word(buffer, word, console):
    """Process a single word: add to buffer, detect blocks, render complete blocks."""
    # Add word to buffer
    buffer += word

    # Try to detect a complete block
    block_result = detect_complete_block(buffer)

    if block_result:
        block_text, block_type, line_begin, line_end = block_result
        
        # Render the complete block
        console.print(Markdown(block_text))
        
        # Remove the rendered block from buffer using line numbers
        lines = buffer.split('\n')
        remaining_lines = lines[line_end:]
        buffer = '\n'.join(remaining_lines)

    return buffer


def stream_markdown_words(console, words, delay=0.1, window_fraction=0.4):
    """
    Stream markdown words (character chunks) with true incremental parsing.
    Detects complete blocks as words arrive and renders them immediately.

    Args:
        console: Rich Console instance
        words: List of character chunks (simulating LLM tokens)
        delay: Delay between words in seconds
        window_fraction: Fraction of terminal height to use for sliding window
    """
    # Calculate window size based on terminal height
    terminal_height = console.size.height
    window_lines = max(8, int(terminal_height * window_fraction))  # Minimum 8 lines

    # Create a console with highlighting disabled for streaming
    plain_console = Console(highlight=False)

    buffer = ""

    with Live(console=plain_console, refresh_per_second=20,
              vertical_overflow="ellipsis") as live:

        for word in words:
            # Process the word (add to buffer, detect blocks, render complete blocks)
            buffer = process_word(buffer, word, console)

            # Stream the remaining buffer content
            if buffer.strip():
                display_text = create_display_text(buffer, window_lines, console)
                live.update(display_text)

                time.sleep(delay)

        # Final cleanup - render any remaining content
        time.sleep(0.5)
        if buffer.strip():
            try:
                live.update(Markdown(buffer))
            except (IndexError, ValueError, TypeError):
                live.update(buffer)


def main():
    console = Console()

    # Load markdown content from external file
    with open('sample_markdown.md', 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # Server simulation: break content into character chunks (LLM tokens)
    words = break_into_words(markdown_text, chunk_size=3)

    print(f"Simulating server sending {len(words)} words to client...")
    print("=" * 60)

    # Client simulation: stream words with true incremental parsing
    stream_markdown_words(console, words, delay=0.1, window_fraction=0.1)

if __name__ == "__main__":
    main()
