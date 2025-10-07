#!/usr/bin/env python3
"""
Debug script to understand newline handling differences.
"""

from rich.console import Console
from rich.markdown import Markdown
from markdown_it import MarkdownIt

def analyze_markdown_structure():
    """Analyze the markdown structure to understand newline handling."""
    
    # Load the markdown content
    with open('sample_markdown.md', 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    
    # Parse with markdown-it
    md = MarkdownIt().enable("strikethrough").enable("table")
    tokens = md.parse(markdown_text)
    
    lines = markdown_text.split('\n')
    
    print("=== MARKDOWN STRUCTURE ANALYSIS ===")
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
    
    for i, token in enumerate(top_level_tokens):
        line_begin, line_end = token.map
        block_lines = lines[line_begin:line_end]
        block_text = '\n'.join(block_lines)
        
        print(f"Block {i+1}: {token.type} (lines {line_begin+1}-{line_end})")
        print(f"  Content preview: {repr(block_text[:50])}...")
        
        # Show the lines around this block
        print(f"  Lines around block:")
        start_line = max(0, line_begin - 1)
        end_line = min(len(lines), line_end + 2)
        
        for line_num in range(start_line, end_line):
            marker = ">>> " if line_begin <= line_num < line_end else "    "
            line_content = lines[line_num]
            print(f"  {marker}{line_num+1:3d}: {repr(line_content)}")
        
        print()

if __name__ == "__main__":
    analyze_markdown_structure()