#!/usr/bin/env python3
"""
Test script for computer.web.answer() function.

Usage:
    python test_search_answer.py [query] [backend]

Examples:
    python test_search_answer.py "What is machine learning?"
    python test_search_answer.py "What is Python?" tavily
    python test_search_answer.py "What is Python?" linkup

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
        query = "What is the name of the project that won the 2008 ACM Software System Award?"

    # Get backend from command line if provided
    backend = None
    if len(sys.argv) > 2:
        backend = sys.argv[2].lower()

    # Create a minimal mock interpreter
    mock_interpreter = Mock()

    # Create Computer instance
    computer = Computer(mock_interpreter)

    # Define all backends to test for answer method
    all_backends = ["tavily", "linkup"]

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
        result = computer.web.answer(query, backend=backend_name)
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
            answer_preview = result.get("answer", "")[:100]
            sources_count = len(result.get("sources", []))
            print(f"✓ {backend_name}: SUCCESS - {sources_count} sources, answer: {answer_preview}...")


if __name__ == "__main__":
    main()
