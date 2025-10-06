#!/usr/bin/env python3
"""
Example script demonstrating how to print complex markdown-formatted text
using the rich library. This includes code blocks, tables, nested elements,
and various markdown features.
"""

from rich.console import Console
from rich.markdown import Markdown

def main():
    console = Console()

    # Load markdown content from external file
    with open('sample_markdown.md', 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # Print the markdown using rich
    console.print(Markdown(markdown_text))

if __name__ == "__main__":
    main()
