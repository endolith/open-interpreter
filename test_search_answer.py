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


def test_answer(query, backend=None):
    """Test the answer function with a given query and optional backend."""
    print(f"\n{'='*60}")
    print(f"Testing: {query}")
    if backend:
        print(f"Backend: {backend}")
    else:
        print("Backend: auto-select")
    print(f"{'='*60}\n")

    # Create a minimal mock interpreter
    mock_interpreter = Mock()

    # Create Computer instance
    computer = Computer(mock_interpreter)

    # Test the answer function
    try:
        if backend:
            result = computer.search.answer(query, backend=backend)
        else:
            result = computer.search.answer(query)

        # Check if it's an error response
        if "error" in result:
            print("❌ Error Response:")
            print(f"  Error: {result.get('error')}")
            print(f"  Message: {result.get('message')}")
            print(f"  Alternative: {result.get('alternative')}")
            if "debug" in result:
                print(f"  Debug: {result.get('debug')}")
            return False

        # Success response
        print("✅ Success Response:")
        print(f"\nAnswer:\n{result.get('answer', '')}\n")

        sources = result.get('sources', [])
        print(f"Sources ({len(sources)}):")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source.get('title', 'No title')}")
            print(f"     URL: {source.get('url', 'No URL')}")
            if source.get('snippet'):
                print(f"     Snippet: {source.get('snippet', '')[:100]}...")
            print()

        return True

    except ValueError as e:
        print(f"❌ ValueError (programming error): {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


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

    # Run the test
    success = test_answer(query, backend)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

