#!/usr/bin/env python3
"""
Test our streaming implementation block by block.
"""

from rich.console import Console
from rich.markdown import Markdown
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

def test_streaming_blocks():
    """Test our streaming implementation block by block."""
    
    console = Console()
    
    # Load the markdown content
    with open('sample_markdown.md', 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    
    print("=== STREAMING IMPLEMENTATION BLOCKS ===")
    
    # Simulate the first few blocks
    buffer = markdown_text
    block_count = 0
    
    while block_count < 5:  # Only show first 5 blocks
        block_result = detect_complete_block(buffer)
        if not block_result:
            break
            
        block_text, block_type, line_begin, line_end = block_result
        block_count += 1
        
        print(f"\n--- Block {block_count}: {block_type} (lines {line_begin+1}-{line_end}) ---")
        print(f"Block text: {repr(block_text)}")
        console.print(Markdown(block_text))
        
        # Remove the block from buffer
        lines = buffer.split('\n')
        remaining_lines = lines[line_end:]
        buffer = '\n'.join(remaining_lines)

if __name__ == "__main__":
    test_streaming_blocks()