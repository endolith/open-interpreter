#!/usr/bin/env python3
"""
Debug script to understand buffer management.
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
            
            # If we found empty lines, include them in the block
            if extended_line_end > line_end:
                block_lines = lines[line_begin:extended_line_end]
                block_text = '\n'.join(block_lines)
                if block_text.strip():
                    return block_text, first_token.type, line_begin, extended_line_end
            else:
                # No empty lines to include, use original block
                block_lines = lines[line_begin:line_end]
                block_text = '\n'.join(block_lines)
                if block_text.strip():
                    return block_text, first_token.type, line_begin, line_end

        return None
    except Exception:
        return None

def debug_buffer_management():
    """Debug the buffer management logic."""
    
    # Load the markdown content
    with open('sample_markdown.md', 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    
    lines = markdown_text.split('\n')
    
    print("=== BUFFER MANAGEMENT DEBUG ===")
    print(f"Total lines: {len(lines)}")
    print()
    
    # Simulate the first few blocks
    buffer = markdown_text
    block_count = 0
    
    while True:
        block_result = detect_complete_block(buffer)
        if not block_result:
            break
            
        block_text, block_type, line_begin, line_end = block_result
        block_count += 1
        
        print(f"Block {block_count}: {block_type} (lines {line_begin+1}-{line_end})")
        print(f"  Block text: {repr(block_text[:50])}...")
        
        # Show what happens when we remove the block
        lines_before = buffer.split('\n')
        remaining_lines = lines_before[line_end:]
        buffer_after = '\n'.join(remaining_lines)
        
        print(f"  Lines before: {len(lines_before)}")
        print(f"  Lines after: {len(remaining_lines)}")
        print(f"  Buffer after: {repr(buffer_after[:100])}...")
        
        buffer = buffer_after
        
        if block_count >= 5:  # Only show first 5 blocks
            break
        
        print()

if __name__ == "__main__":
    debug_buffer_management()