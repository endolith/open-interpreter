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

    # Print the markdown using rich
    console.print(Markdown(markdown_text))

if __name__ == "__main__":
    main()
