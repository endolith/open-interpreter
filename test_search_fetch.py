#!/usr/bin/env python3
"""
Test script for computer.search.fetch() function.

Usage:
    python test_search_fetch.py [url] [backend]

Examples:
    python test_search_fetch.py "https://example.com"
    python test_search_fetch.py "https://example.com" serper
    python test_search_fetch.py "https://example.com" linkup
    python test_search_fetch.py "https://example.com" tavily

If no backend is specified, tests all available backends.
"""

import sys
from unittest.mock import Mock

from interpreter.core.computer.computer import Computer


def main():
    """Main test function."""
    # Get URL from command line or use default
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://example.com"

    # Get backend from command line if provided
    backend = None
    if len(sys.argv) > 2:
        backend = sys.argv[2].lower()

    # Create a minimal mock interpreter
    mock_interpreter = Mock()

    # Create Computer instance
    computer = Computer(mock_interpreter)

    # Define all backends to test for fetch method
    all_backends = ["serper", "linkup", "tavily"]

    if backend:
        # Test only the specified backend
        backends_to_test = [backend]
    else:
        # Test all backends
        backends_to_test = all_backends

    print(f"\nTesting URL: '{url}'")
    print(f"Backends to test: {', '.join(backends_to_test)}\n")

    results = {}
    for backend_name in backends_to_test:
        print("\n" + "="*60)
        print(f"Testing backend: {backend_name}")
        print("="*60)

        # Call the method - it will print its own output
        result = computer.search.fetch(url, backend=backend_name)
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
            if "results" in result:
                # Tavily returns multiple results
                results_count = len(result.get("results", []))
                first_title = result.get("results", [{}])[0].get("title", "N/A") if results_count > 0 else "N/A"
                content_preview = result.get("results", [{}])[0].get("content", "")[:100] if results_count > 0 else ""
                print(f"✓ {backend_name}: SUCCESS - {results_count} result(s), first: {first_title}")
                if content_preview:
                    print(f"   Content preview: {content_preview}...")
            else:
                # Single page result (serper, linkup)
                title = result.get("title", "N/A")
                content_preview = result.get("content", "")[:100] if result.get("content") else ""
                print(f"✓ {backend_name}: SUCCESS - Title: {title}")
                if content_preview:
                    print(f"   Content preview: {content_preview}...")


if __name__ == "__main__":
    main()

