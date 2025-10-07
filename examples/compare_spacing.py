#!/usr/bin/env python3
"""
Compare the spacing between our streaming output and the ground truth.
"""

from rich.console import Console
from rich.markdown import Markdown

def compare_spacing():
    """Compare the spacing between our streaming output and the ground truth."""
    
    console = Console()
    
    # Test the exact spacing from the ground truth
    print("=== GROUND TRUTH SPACING ===")
    
    # Block 1: Heading
    block1 = "# Rich Markdown Example\n"
    console.print(Markdown(block1))
    
    # Block 2: Paragraph  
    block2 = "This is a comprehensive example of markdown formatting using the **rich** library.\n"
    console.print(Markdown(block2))
    
    # Block 3: Heading
    block3 = "## Features Demonstrated\n"
    console.print(Markdown(block3))
    
    # Block 4: List
    block4 = """- **Bold text** and *italic text*
- `inline code` and code blocks
- Tables with various formatting
- Nested quotes and lists
- Links and images
- Horizontal rules
"""
    console.print(Markdown(block4))
    
    # Block 5: Heading
    block5 = "### Code Examples\n"
    console.print(Markdown(block5))
    
    # Block 6: Paragraph
    block6 = "Here's a simple Python function:\n"
    console.print(Markdown(block6))

if __name__ == "__main__":
    compare_spacing()