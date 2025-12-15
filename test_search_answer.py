#!/usr/bin/env python3
"""
Test script for computer.search.answer() function.

Usage:
    python test_search_answer.py [query] [backend]

Examples:
    python test_search_answer.py "What is machine learning?"
    python test_search_answer.py "What is Python?" tavily
    python test_search_answer.py "What is Python?" linkup
"""

import sys
from unittest.mock import Mock

from interpreter.core.computer.computer import Computer


def main():
    """Main test function."""
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "What is the name of the project that won the 2008 ACM Software System Award?"

    # Get backend from command line if provided
    backend = None
    if len(sys.argv) > 2:
        backend = sys.argv[2].lower()

    # Create a minimal mock interpreter
    mock_interpreter = Mock()

    # Create Computer instance
    computer = Computer(mock_interpreter)

    # Call the method - it will print its own output
    if backend:
        result = computer.search.answer(query, backend=backend)
    else:
        result = computer.search.answer(query)

    # Show the raw return value
    print("\n" + "="*60)
    print("Returned object:")
    print("="*60)
    print(repr(result))


if __name__ == "__main__":
    main()

