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
    Detect complete blocks by finding when a new top-level block starts.
    Returns (complete_block_text, block_type, line_begin, line_end) when a complete block is found.
    """
    try:
        md = MarkdownIt().enable("strikethrough").enable("table")
        tokens = md.parse(markdown_text)

        lines = markdown_text.split('\n')

        # Find all top-level tokens (nesting level 1 or self-closing)
        top_level_tokens = []
        nesting_level = 0

        for token in tokens:
            if not token.block or token.type == 'inline':
                continue

            if token.nesting == 1:  # Opening token
                nesting_level += 1
                if nesting_level == 1:  # Top-level opening
                    top_level_tokens.append(token)

            elif token.nesting == -1:  # Closing token
                nesting_level -= 1

            elif token.nesting == 0:  # Self-closing token (like fence)
                if nesting_level == 0:  # Only if we're at top level
                    top_level_tokens.append(token)

        # If we have at least 2 top-level blocks, the first one is complete
        if len(top_level_tokens) >= 2:
            first_token = top_level_tokens[0]
            line_begin, line_end = first_token.map
            
            # Include trailing empty lines to preserve spacing between blocks
            # Look for empty lines immediately after the block
            extended_line_end = line_end
            while extended_line_end < len(lines) and lines[extended_line_end].strip() == '':
                extended_line_end += 1
            
            # Always include the extended block to preserve spacing
            block_lines = lines[line_begin:extended_line_end]
            block_text = '\n'.join(block_lines)
            if block_text.strip():
                return block_text, first_token.type, line_begin, extended_line_end

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
    """Process a single word: add to buffer, detect blocks, return complete blocks."""
    # Add word to buffer
    buffer += word

    # Try to detect a complete block
    block_result = detect_complete_block(buffer)

    if block_result:
        block_text, block_type, line_begin, line_end = block_result

        # Remove the rendered block from buffer using line numbers
        lines = buffer.split('\n')
        
        # Preserve empty lines that follow the block to maintain proper spacing
        # The block ends at line_end, so we need to keep everything from line_end onwards
        # This includes any empty lines that exist between blocks in the original markdown
        remaining_lines = lines[line_end:]
        buffer = '\n'.join(remaining_lines)

        # Return the complete block to be rendered
        return buffer, Markdown(block_text)

    return buffer, None


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

    buffer = ""
    prev_element_new_line = False

    with Live(console=console, refresh_per_second=20,
              vertical_overflow="ellipsis") as live:

        for word in words:
            # Process the word (add to buffer, detect blocks, return complete blocks)
            buffer, complete_block = process_word(buffer, word, console)

            # If we have a complete block, render it and clear the live window
            if complete_block:
                # Clear the live window and render the complete block
                # Rich handles spacing between blocks automatically when rendered together
                live.update("")
                console.print(complete_block)

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
