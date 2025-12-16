#!/usr/bin/env python3
"""
Test script for computer.search.search() function.

Usage:
    python test_search.py [query] [backend]

Examples:
    python test_search.py "machine learning tutorials"
    python test_search.py "Python programming" brave
    python test_search.py "latest AI news" tavily
    python test_search.py "AI research" linkup
    python test_search.py "tech news" serpapi
    python test_search.py "coding tips" serper

If no backend is specified, tests all available backends.
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
        query = "machine learning tutorials"

    # Get backend from command line if provided
    backend = None
    if len(sys.argv) > 2:
        backend = sys.argv[2].lower()

    # Create a minimal mock interpreter
    mock_interpreter = Mock()

    # Create Computer instance
    computer = Computer(mock_interpreter)

    # Define all backends to test for search method
    all_backends = ["brave", "tavily", "linkup", "serpapi", "serper"]

    if backend:
        # Test only the specified backend
        backends_to_test = [backend]
    else:
        # Test all backends
        backends_to_test = all_backends

    print(f"\nTesting query: '{query}'")
    print(f"Backends to test: {', '.join(backends_to_test)}\n")

    results = {}
    for backend_name in backends_to_test:
        print("\n" + "="*60)
        print(f"Testing backend: {backend_name}")
        print("="*60)

        # Call the method - it will print its own output
        result = computer.search.search(query, backend=backend_name)
        results[backend_name] = result

        # Show the raw return value
        print("\n" + "-"*60)
        print("Returned object:")
        print("-"*60)
        print(repr(result))

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for backend_name, result in results.items():
        if "error" in result:
            print(f"❌ {backend_name}: FAILED - {result['error']}")
        else:
            results_count = len(result.get("results", []))
            first_title = result.get("results", [{}])[0].get("title", "N/A") if results_count > 0 else "N/A"
            print(f"✓ {backend_name}: SUCCESS - {results_count} results, first: {first_title}")


if __name__ == "__main__":
    main()
