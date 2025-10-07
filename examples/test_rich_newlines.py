#!/usr/bin/env python3
"""
Test how Rich handles newlines in markdown blocks.
"""

from rich.console import Console
from rich.markdown import Markdown

def test_rich_newlines():
    """Test how Rich handles newlines in markdown blocks."""
    
    console = Console()
    
    # Test 1: Block with trailing newline
    print("=== Test 1: Block with trailing newline ===")
    block1 = "# Heading\n"
    console.print(Markdown(block1))
    
    # Test 2: Block without trailing newline
    print("=== Test 2: Block without trailing newline ===")
    block2 = "# Heading"
    console.print(Markdown(block2))
    
    # Test 3: Two blocks with newline between them
    print("=== Test 3: Two blocks with newline between them ===")
    block3 = "# Heading\n\n## Subheading"
    console.print(Markdown(block3))
    
    # Test 4: Two blocks without newline between them
    print("=== Test 4: Two blocks without newline between them ===")
    block4 = "# Heading\n## Subheading"
    console.print(Markdown(block4))

if __name__ == "__main__":
    test_rich_newlines()