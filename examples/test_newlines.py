#!/usr/bin/env python3
"""
Test how Rich and markdown-it-py handle different newline patterns
"""

from rich.console import Console
from rich.markdown import Markdown
from markdown_it import MarkdownIt


def test_newline_patterns():
    """Test different newline patterns between blocks."""

    test_cases = [
        ("Single newline", "# Heading\nThis is a paragraph."),
        ("Double newline", "# Heading\n\nThis is a paragraph."),
        ("Triple newline", "# Heading\n\n\nThis is a paragraph."),
        ("Quad newline", "# Heading\n\n\n\nThis is a paragraph."),
        ("Mixed blocks", "# Heading\n\nThis is paragraph 1.\n\nThis is paragraph 2.\n\n## Another Heading\n\nMore text."),
    ]

    console = Console()

    for name, text in test_cases:
        print(f"\n=== {name} ===")
        print(f"Input: {repr(text)}")

        # Test how markdown-it-py parses it
        print("\nMarkdown-it-py parsing:")
        md = MarkdownIt().enable("strikethrough").enable("table")
        tokens = md.parse(text)

        lines = text.split('\n')
        block_tokens = []
        for token in tokens:
            if token.block and token.map and (token.nesting == 1 or token.nesting == 0) and token.type != 'inline':
                line_begin, line_end = token.map
                block_lines = lines[line_begin:line_end]
                block_text = '\n'.join(block_lines)
                if block_text.strip():
                    block_tokens.append((block_text, token.type, line_begin, line_end))

        for i, (block_text, block_type, line_begin, line_end) in enumerate(block_tokens):
            print(f"  Block {i}: {block_type} (lines {line_begin}-{line_end}) -> {repr(block_text)}")

        # Test how Rich renders it
        print("\nRich rendering:")
        console.print(Markdown(text))
        print("-" * 50)


def test_buffer_removal():
    """Test buffer removal logic with different newline patterns."""

    test_cases = [
        "# Heading\n\nThis is a paragraph.",
        "# Heading\n\n\nThis is a paragraph.",
        "# Heading\n\n\n\nThis is a paragraph.",
    ]

    for text in test_cases:
        print(f"\n=== Testing buffer removal for: {repr(text)} ===")

        # Simulate the buffer removal logic
        md = MarkdownIt().enable("strikethrough").enable("table")
        tokens = md.parse(text)

        lines = text.split('\n')
        block_tokens = []
        for token in tokens:
            if token.block and token.map and (token.nesting == 1 or token.nesting == 0) and token.type != 'inline':
                line_begin, line_end = token.map
                block_lines = lines[line_begin:line_end]
                block_text = '\n'.join(block_lines)
                if block_text.strip():
                    block_tokens.append((block_text, token.type, line_begin, line_end))

        if len(block_tokens) >= 2:
            first_block_text, first_block_type, line_begin, line_end = block_tokens[0]
            print(f"First complete block: {first_block_type} -> {repr(first_block_text)}")
            print(f"Line range: {line_begin} to {line_end}")

            # Test buffer removal
            remaining_lines = lines[line_end:]
            remaining_buffer = '\n'.join(remaining_lines)
            print(f"Original buffer: {repr(text)}")
            print(f"Remaining buffer: {repr(remaining_buffer)}")
            print(f"Removed lines: {lines[line_begin:line_end]}")


if __name__ == "__main__":
    print("=== Testing Newline Patterns ===")
    test_newline_patterns()

    print("\n\n=== Testing Buffer Removal Logic ===")
    test_buffer_removal()
