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


def stream_markdown_blocks(console, markdown_text, chunk_size=10, delay=0.1):
    """
    Stream markdown text block by block, creating a new Live object for each block.
    """
    # Hardcoded block boundaries - split the text into independent blocks
    blocks = [
        "# Rich Markdown Example",

        "This is a comprehensive example of markdown formatting using the **rich** library.",

        "## Features Demonstrated\n\n- **Bold text** and *italic text*\n- `inline code` and code blocks\n- Tables with various formatting\n- Nested quotes and lists\n- Links and images\n- Horizontal rules",

        "### Code Examples\n\nHere's a simple Python function:",

        "```python\ndef fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Example usage\nfor i in range(10):\n    print(f\"F({i}) = {fibonacci(i)}\")\n```",

        "And here's some JavaScript code:",

        "```javascript\n// Async function example\nasync function fetchUserData(userId) {\n    try {\n        const response = await fetch(`/api/users/${userId}`);\n        const userData = await response.json();\n        return userData;\n    } catch (error) {\n        console.error('Error fetching user data:', error);\n        throw error;\n    }\n}\n\n// Usage\nfetchUserData(123)\n    .then(data => console.log('User data:', data))\n    .catch(error => console.error('Failed to fetch user:', error));\n```",

        "### Nested Code Blocks\n\nSometimes you need to show code that contains other code blocks:",

        "````markdown\nHere's how to create a code block in markdown:\n\n```python\nprint(\"Hello, World!\")\n```\n\nThe syntax is three backticks followed by the language name.\n````",

        "### Tables\n\n| Feature | Status | Priority | Notes |\n|---------|--------|----------|-------|\n| Authentication | ✅ Complete | High | Uses JWT tokens |\n| Database | 🔄 In Progress | High | PostgreSQL implementation |\n| API Documentation | ❌ Not Started | Medium | Will use OpenAPI |\n| Testing | ✅ Complete | High | 95% code coverage |\n| Deployment | 🔄 In Progress | Medium | Docker containers |",

        "### Complex Nested Lists\n\n1. **Frontend Development**\n   - React Components\n     - Functional components\n     - Class components\n     - Hooks usage\n   - State Management\n     - Redux\n     - Context API\n     - Local state\n   - Styling\n     - CSS Modules\n     - Styled Components\n     - Tailwind CSS\n\n2. **Backend Development**\n   - API Design\n     - REST endpoints\n     - GraphQL queries\n     - WebSocket connections\n   - Database\n     - Schema design\n     - Migrations\n     - Query optimization\n   - Authentication\n     - JWT tokens\n     - OAuth integration\n     - Role-based access",

        "### Blockquotes\n\n> This is a simple blockquote.\n\n> This is a blockquote with **bold text** and `inline code`.\n>\n> It can span multiple lines and contain various formatting.\n\n> #### Nested Quote\n>\n> > This is a nested quote inside another quote.\n> >\n> > It demonstrates how quotes can be nested:\n> >\n> > ```python\n> > # Code can even be inside quotes\n> > def example():\n> >     return \"Hello from nested quote!\"\n> > ```",

        "### Links and References\n\n- [Rich Documentation](https://rich.readthedocs.io/)\n- [Markdown Guide](https://www.markdownguide.org/)\n- [Python.org](https://www.python.org/)",

        "### Mathematical Expressions\n\nWhile rich doesn't support LaTeX math, you can still show mathematical concepts:\n\n- The quadratic formula: `x = (-b ± √(b² - 4ac)) / 2a`\n- Euler's identity: `e^(iπ) + 1 = 0`\n- The golden ratio: `φ = (1 + √5) / 2 ≈ 1.618`",

        "### Task Lists\n\n- [x] Set up development environment\n- [x] Create basic project structure\n- [x] Implement core functionality\n- [ ] Write comprehensive tests\n- [ ] Add documentation\n- [ ] Deploy to production\n- [ ] Monitor performance",

        "### Horizontal Rule\n\n---",

        "### Final Notes\n\nThis example demonstrates the power of the `rich` library for displaying\ncomplex markdown content in the terminal. The library handles:\n\n- Syntax highlighting for code blocks\n- Proper table formatting\n- Nested list indentation\n- Blockquote styling\n- Link formatting\n- And much more!\n\n> **Tip**: You can customize the appearance by modifying the Console\n> configuration or using different themes.\n\n```python\n# Example of custom console configuration\nfrom rich.console import Console\nfrom rich.theme import Theme\n\ncustom_theme = Theme({\n    \"markdown.code\": \"bold blue\",\n    \"markdown.heading\": \"bold magenta\",\n    \"markdown.strong\": \"bold red\"\n})\n\nconsole = Console(theme=custom_theme)\n```\n\n---\n\n*End of example*"
    ]

    for block in blocks:
        # Create a new Live object for each block
        with Live(console=console, refresh_per_second=10,
                  vertical_overflow="visible") as live:

            # Stream the current block character by character
            block_accumulated = ""
            for i in range(0, len(block), chunk_size):
                chunk = block[i:i + chunk_size]
                block_accumulated += chunk

                # Update with just the current block content
                try:
                    live.update(Markdown(block_accumulated))
                except (IndexError, ValueError, TypeError):
                    # If markdown parsing fails, just skip this update
                    pass

                time.sleep(delay)

        # Block is complete, move to next (no need to print again)


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

    # Stream the markdown text block by block
    stream_markdown_blocks(console, markdown_text, chunk_size=10, delay=0.1)


if __name__ == "__main__":
    main()
