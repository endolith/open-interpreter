#!/usr/bin/env python3
"""
Debug script to understand block detection and newline handling.
"""

from rich.console import Console
from rich.markdown import Markdown
from markdown_it import MarkdownIt

def debug_block_detection():
    """Debug the block detection logic."""
    
    # Load the markdown content
    with open('sample_markdown.md', 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    
    # Parse with markdown-it
    md = MarkdownIt().enable("strikethrough").enable("table")
    tokens = md.parse(markdown_text)
    
    lines = markdown_text.split('\n')
    
    print("=== BLOCK DETECTION DEBUG ===")
    print(f"Total lines: {len(lines)}")
    print()
    
    # Find all top-level tokens
    top_level_tokens = []
    nesting_level = 0
    
    for i, token in enumerate(tokens):
        if not token.block or token.type == 'inline':
            continue
            
        if token.nesting == 1:  # Opening token
            nesting_level += 1
            if nesting_level == 1:  # Top-level opening
                top_level_tokens.append(token)
        elif token.nesting == -1:  # Closing token
            nesting_level -= 1
        elif token.nesting == 0:  # Self-closing token
            if nesting_level == 0:  # Only if we're at top level
                top_level_tokens.append(token)
    
    print(f"Found {len(top_level_tokens)} top-level blocks:")
    print()
    
    # Test the first few blocks
    for i in range(min(5, len(top_level_tokens))):
        token = top_level_tokens[i]
        line_begin, line_end = token.map
        
        print(f"Block {i+1}: {token.type} (lines {line_begin+1}-{line_end})")
        
        # Show the original block
        original_block = lines[line_begin:line_end]
        print(f"  Original block: {repr('\n'.join(original_block))}")
        
        # Show what happens with extended line end
        extended_line_end = line_end
        while extended_line_end < len(lines) and lines[extended_line_end].strip() == '':
            extended_line_end += 1
        
        if extended_line_end > line_end:
            extended_block = lines[line_begin:extended_line_end]
            print(f"  Extended block: {repr('\n'.join(extended_block))}")
            print(f"  Extended by {extended_line_end - line_end} lines")
        else:
            print(f"  No extension needed")
        
        print()

if __name__ == "__main__":
    debug_block_detection()