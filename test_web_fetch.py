#!/usr/bin/env python3
"""
Test script for computer.web.fetch() function.

Usage:
    python test_search_fetch.py [url] [backend]
    python test_search_fetch.py --all [backend]  # Run comprehensive test suite

Examples:
    python test_search_fetch.py "https://example.com"
    python test_search_fetch.py "https://example.com" serper
    python test_search_fetch.py --all  # Test all URL types
    python test_search_fetch.py --all serper  # Test all URL types with specific backend

If no backend is specified, tests all available backends.
"""

import sys
from unittest.mock import Mock

from interpreter.core.toolbox.toolbox import Toolbox
from interpreter.core.toolbox.web.web import WebToolboxError


# Comprehensive test URLs covering different content types
TEST_URLS = {
    "Basic Page": "https://example.com",
    "Wikipedia": "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "Dynamic Content": "https://www.nytimes.com/",
    "JavaScript SPA": "https://reactjs.org/",
    "PDF": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "Redirect": "http://httpbin.org/redirect-to?url=https://example.com",
    "404 Error": "https://httpbin.org/status/404",
    "JSON API": "https://api.github.com/repos/python/cpython",
}


def run_single_url(toolbox, url, backend_name):
    """Run fetch for one URL with a specific backend (CLI helper, not a pytest test)."""
    print("\n" + "="*60)
    print(f"Testing backend: {backend_name}")
    print("="*60)

    try:
        result = toolbox.web.fetch(url, backend=backend_name)
    except WebToolboxError as e:
        result = {"error": str(e)}

    print("\n" + "-"*60)
    print("Returned object (truncated):")
    print("-"*60)
    result_str = repr(result)
    if len(result_str) > 500:
        print(result_str[:500] + "... [truncated]")
    else:
        print(result_str)

    return result


def format_result_summary(backend_name, result):
    """Format a single result for summary output."""
    if "error" in result:
        return f"❌ {backend_name}: FAILED - {result['error']}"
    else:
        if "results" in result:
            # Tavily returns multiple results
            results_count = len(result.get("results", []))
            first_title = result.get("results", [{}])[0].get("title", "N/A") if results_count > 0 else "N/A"
            content_preview = result.get("results", [{}])[0].get("content", "")[:100] if results_count > 0 else ""
            summary = f"✓ {backend_name}: SUCCESS - {results_count} result(s), first: {first_title}"
            if content_preview:
                summary += f"\n   Content preview: {content_preview}..."
            return summary
        else:
            # Single page result (serper, linkup)
            title = result.get("title", "N/A")
            content_preview = result.get("content", "")[:100] if result.get("content") else ""
            summary = f"✓ {backend_name}: SUCCESS - Title: {title}"
            if content_preview:
                summary += f"\n   Content preview: {content_preview}..."
            return summary


def main():
    """Main test function."""
    # Check for --all flag
    test_all = "--all" in sys.argv
    if test_all:
        sys.argv.remove("--all")

    # Get URL from command line or use default
    if len(sys.argv) > 1:
        url = sys.argv[1]
        test_urls = {url: url}  # Single URL test
    else:
        if test_all:
            test_urls = TEST_URLS
        else:
            # Use Wikipedia as default - works well with all backends
            test_urls = {"Default": "https://en.wikipedia.org/wiki/Python_(programming_language)"}

    # Get backend from command line if provided
    backend = None
    if len(sys.argv) > 2:
        backend = sys.argv[2].lower()

    # Create a minimal mock interpreter
    mock_interpreter = Mock()

    # Create Toolbox instance
    toolbox = Toolbox(mock_interpreter)

    # Define all backends to test for fetch method
    all_backends = ["serper", "linkup", "tavily"]

    if backend:
        # Test only the specified backend
        backends_to_test = [backend]
    else:
        # Test all backends
        backends_to_test = all_backends

    # Run tests
    if test_all or len(test_urls) > 1:
        # Comprehensive test suite
        print("\n" + "="*60)
        print("COMPREHENSIVE FETCH TEST SUITE")
        print("="*60)
        print(f"Testing {len(test_urls)} URL types")
        print(f"Backends to test: {', '.join(backends_to_test)}\n")

        all_results = {}
        for url_type, url in test_urls.items():
            print("\n" + "="*80)
            print(f"URL Type: {url_type}")
            print(f"URL: {url}")
            print("="*80)

            url_results = {}
            for backend_name in backends_to_test:
                result = run_single_url(toolbox, url, backend_name)
                url_results[backend_name] = result

            all_results[url_type] = url_results

        # Summary
        print("\n\n" + "="*80)
        print("COMPREHENSIVE SUMMARY")
        print("="*80)
        for url_type, url_results in all_results.items():
            print(f"\n{url_type} ({TEST_URLS.get(url_type, url_type)}):")
            for backend_name, result in url_results.items():
                print(f"  {format_result_summary(backend_name, result)}")
    else:
        # Single URL test
        url = list(test_urls.values())[0]
        print(f"\nTesting URL: '{url}'")
        print(f"Backends to test: {', '.join(backends_to_test)}\n")

        results = {}
        for backend_name in backends_to_test:
            result = run_single_url(toolbox, url, backend_name)
            results[backend_name] = result

        # Summary
        print("\n\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for backend_name, result in results.items():
            print(format_result_summary(backend_name, result))


if __name__ == "__main__":
    main()
