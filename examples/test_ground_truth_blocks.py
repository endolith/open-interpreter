#!/usr/bin/env python3
"""
Test rendering the ground truth block by block with proper spacing.
"""

from rich.console import Console
from rich.markdown import Markdown

def test_ground_truth_blocks():
    """Test rendering the ground truth block by block with proper spacing."""
    
    console = Console()
    
    # Load the markdown content
    with open('sample_markdown.md', 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    
    # Parse with markdown-it to get blocks
    from markdown_it import MarkdownIt
    md = MarkdownIt().enable("strikethrough").enable("table")
    tokens = md.parse(markdown_text)
    
    lines = markdown_text.split('\n')
    
    # Find all top-level tokens
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
        elif token.nesting == 0:  # Self-closing token
            if nesting_level == 0:  # Only if we're at top level
                top_level_tokens.append(token)
    
    print("=== GROUND TRUTH BLOCK BY BLOCK WITH PROPER SPACING ===")
    
    # Test the first few blocks with proper spacing
    for i in range(min(5, len(top_level_tokens))):
        token = top_level_tokens[i]
        line_begin, line_end = token.map
        
        print(f"\n--- Block {i+1}: {token.type} (lines {line_begin+1}-{line_end}) ---")
        
        # Get the block with proper spacing - include trailing empty lines
        extended_line_end = line_end
        while extended_line_end < len(lines) and lines[extended_line_end].strip() == '':
            extended_line_end += 1
        
        block_lines = lines[line_begin:extended_line_end]
        block_text = '\n'.join(block_lines)
        
        print(f"Block text: {repr(block_text)}")
        console.print(Markdown(block_text))

if __name__ == "__main__":
    test_ground_truth_blocks()