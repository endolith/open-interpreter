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
from markdown_it import MarkdownIt


def parse_markdown_into_blocks(markdown_text):
    """
    Parse markdown into blocks using markdown-it-py's token stream.
    Extract complete blocks by processing tokens in order and grouping related tokens.
    """
    md = MarkdownIt().enable("strikethrough").enable("table")
    tokens = md.parse(markdown_text)

    lines = markdown_text.split('\n')
    blocks = []

    # Process tokens sequentially to build complete blocks
    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Process block-level tokens (both opening and self-closing)
        if token.block and token.map and (token.nesting == 1 or token.nesting == 0):
            line_begin, line_end = token.map

            # Extract the complete block from original source
            block_lines = lines[line_begin:line_end]
            block_text = '\n'.join(block_lines)

            if block_text.strip():
                # Return the original block text along with token type and tag
                blocks.append((block_text, token.type, token.tag))

            # Skip to the end of this block to avoid processing nested content
            if token.nesting == 1:  # Only skip for opening tokens
                nesting_level = 1
                i += 1
                while i < len(tokens) and nesting_level > 0:
                    if tokens[i].nesting == 1:
                        nesting_level += 1
                    elif tokens[i].nesting == -1:
                        nesting_level -= 1
                    i += 1
                continue

        i += 1

    return blocks


def stream_markdown_blocks(console, markdown_text, chunk_size=10, delay=0.1, window_lines=16):
    """
    Stream markdown text block by block using sliding window approach.
    Each block streams with a sliding window, then renders the complete block.
    """
    blocks = parse_markdown_into_blocks(markdown_text)

    prev_element_new_line = False

    for block_text, token_type, token_tag in blocks:
        # Insert a blank line BEFORE this block if the previous element requested a newline
        # Matches Rich's renderer behavior where most elements set new_line=True
        if prev_element_new_line:
            console.print()

        # Create a console with highlighting disabled for streaming
        plain_console = Console(highlight=False)

        # Create a new Live object for each block
        with Live(console=plain_console, refresh_per_second=20,
                  vertical_overflow="ellipsis") as live:

            # Stream the current block with sliding window
            block_accumulated = ""
            for i in range(0, len(block_text), chunk_size):
                chunk = block_text[i:i + chunk_size]
                block_accumulated += chunk

                # Split into lines and keep only the last window_lines
                current_lines = block_accumulated.split('\n')
                if len(current_lines) > window_lines:
                    display_lines = current_lines[-window_lines:]
                    # Add ellipsis to indicate there's more content above
                    display_text = "...\n" + '\n'.join(display_lines)
                else:
                    display_text = '\n'.join(current_lines)

                # Update with plain text (no syntax highlighting during streaming)
                live.update(display_text)
                time.sleep(delay)

            # After streaming is complete, show the full rendered block
            time.sleep(0.2)  # Brief pause before final display
            try:
                live.update(Markdown(block_text))
            except (IndexError, ValueError, TypeError):
                # If markdown parsing fails, show plain text
                live.update(block_text)

        # Decide if a newline should be inserted BEFORE the next element
        # In Rich, HorizontalRule sets new_line=False, everything else True
        prev_element_new_line = (token_type != "hr")


def main():
    console = Console()

    # Complex markdown text with various elements
    markdown_text = """
# Rich Markdown Example

This is a comprehensive example of markdown formatting using the **rich** library.

## Features Demonstrated

- **Bold text** and *italic text*
- `inline code` and code blocks
- Tables with various formatting
- Nested quotes and lists
- Links and images
- Horizontal rules

### Code Examples

Here's a simple Python function:

```python
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
```

And here's some JavaScript code:

```javascript
// Async function example
async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        const userData = await response.json();
        return userData;
    } catch (error) {
        console.error('Error fetching user data:', error);
        throw error;
    }
}

// Usage
fetchUserData(123)
    .then(data => console.log('User data:', data))
    .catch(error => console.error('Failed to fetch user:', error));
```

### Nested Code Blocks

Sometimes you need to show code that contains other code blocks:

````markdown
Here's how to create a code block in markdown:

```python
print("Hello, World!")
```

The syntax is three backticks followed by the language name.
````

### Tables

| Feature | Status | Priority | Notes |
|---------|--------|----------|-------|
| Authentication | ✅ Complete | High | Uses JWT tokens |
| Database | 🔄 In Progress | High | PostgreSQL implementation |
| API Documentation | ❌ Not Started | Medium | Will use OpenAPI |
| Testing | ✅ Complete | High | 95% code coverage |
| Deployment | 🔄 In Progress | Medium | Docker containers |

### Complex Nested Lists

1. **Frontend Development**
   - React Components
     - Functional components
     - Class components
     - Hooks usage
   - State Management
     - Redux
     - Context API
     - Local state
   - Styling
     - CSS Modules
     - Styled Components
     - Tailwind CSS

2. **Backend Development**
   - API Design
     - REST endpoints
     - GraphQL queries
     - WebSocket connections
   - Database
     - Schema design
     - Migrations
     - Query optimization
   - Authentication
     - JWT tokens
     - OAuth integration
     - Role-based access

### Blockquotes

> This is a simple blockquote.

> This is a blockquote with **bold text** and `inline code`.
>
> It can span multiple lines and contain various formatting.

> #### Nested Quote
>
> > This is a nested quote inside another quote.
> >
> > It demonstrates how quotes can be nested:
> >
> > ```python
> > # Code can even be inside quotes
> > def example():
> >     return "Hello from nested quote!"
> > ```

### Links and References

- [Rich Documentation](https://rich.readthedocs.io/)
- [Markdown Guide](https://www.markdownguide.org/)
- [Python.org](https://www.python.org/)

### Mathematical Expressions

While rich doesn't support LaTeX math, you can still show mathematical concepts:

- The quadratic formula: `x = (-b ± √(b² - 4ac)) / 2a`
- Euler's identity: `e^(iπ) + 1 = 0`
- The golden ratio: `φ = (1 + √5) / 2 ≈ 1.618`

### Task Lists

- [x] Set up development environment
- [x] Create basic project structure
- [x] Implement core functionality
- [ ] Write comprehensive tests
- [ ] Add documentation
- [ ] Deploy to production
- [ ] Monitor performance

### Horizontal Rule

---

### Final Notes

This example demonstrates the power of the `rich` library for displaying
complex markdown content in the terminal. The library handles:

- Syntax highlighting for code blocks
- Proper table formatting
- Nested list indentation
- Blockquote styling
- Link formatting
- And much more!

> **Tip**: You can customize the appearance by modifying the Console
> configuration or using different themes.

```python
# Example of custom console configuration
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "markdown.code": "bold blue",
    "markdown.heading": "bold magenta",
    "markdown.strong": "bold red"
})

console = Console(theme=custom_theme)
```

---

*End of example*
"""

    # Stream the markdown text block by block with sliding window
    stream_markdown_blocks(console, markdown_text, chunk_size=10, delay=0.01, window_lines=16)


if __name__ == "__main__":
    main()
