#!/usr/bin/env python3
"""
Comprehensive test script for computer.search.search() with different engines and options.

This script tests various search engines and configurations across all backends.

Usage:
    python test_search_engines.py

Note: You need API keys for the backends you want to test:
    - BRAVE_API_KEY
    - TAVILY_API_KEY
    - LINKUP_API_KEY
    - SERPAPI_API_KEY
    - SERPER_API_KEY
"""

import sys
from unittest.mock import Mock
from interpreter.core.computer.computer import Computer


def test_serpapi_engines():
    """Test SerpApi with different search engines."""
    print("\n" + "="*80)
    print("TESTING SERPAPI - DIFFERENT ENGINES")
    print("="*80)

    mock_interpreter = Mock()
    computer = Computer(mock_interpreter)

    # Test cases: (query, engine, description)
    test_cases = [
        ("how to install python on windows", "google", "Regular Google Search - General web results"),
        ("transformer neural networks attention mechanism", "google_scholar", "Google Scholar - Academic research papers"),
        ("how to make sourdough bread step by step", "youtube", "YouTube Video Search - Video tutorials"),
        ("latest developments in quantum computing 2024", "google_news", "Google News Search - Recent news articles"),
        ("MacBook Pro 16 inch M3 Max 1TB", "google_shopping", "Google Shopping Search - Product listings with prices"),
        ("weather forecast this week", "bing", "Bing Search - Alternative search engine"),
        ("privacy-focused search engine comparison", "duckduckgo", "DuckDuckGo Search - Privacy-focused results"),
    ]

    for query, engine, description in test_cases:
        print(f"\n{'-'*80}")
        print(f"Test: {description}")
        print(f"Query: '{query}', Engine: '{engine}'")
        print(f"{'-'*80}")

        result = computer.search.search(query, backend="serpapi", engine=engine, num=3)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Success - Backend: {result.get('backend')}, Engine: {engine}")
            print(f"  Found {len(result.get('results', []))} results:")
            for i, res in enumerate(result.get('results', [])[:2], 1):
                print(f"    {i}. {res.get('title', 'N/A')}")


def test_serper_types():
    """Test Serper with different search types."""
    print("\n" + "="*80)
    print("TESTING SERPER - DIFFERENT SEARCH TYPES")
    print("="*80)

    mock_interpreter = Mock()
    computer = Computer(mock_interpreter)

    # Test cases: (query, type, description)
    test_cases = [
        ("best practices for REST API design", "search", "Regular Web Search - General web results"),
        ("ChatGPT 4.5 release announcement December 2024", "news", "News Search - Recent news articles"),
        ("React hooks tutorial for beginners", "videos", "Video Search - Video content"),
        ("iPhone 15 Pro Max 256GB price comparison", "shopping", "Shopping Search - Product listings"),
        # NOTE: "scholar" is NOT supported by Serper - use serpapi with engine="google_scholar" instead
    ]

    for query, search_type, description in test_cases:
        print(f"\n{'-'*80}")
        print(f"Test: {description}")
        print(f"Query: '{query}', Type: '{search_type}'")
        print(f"{'-'*80}")

        result = computer.search.search(query, backend="serper", type=search_type, num=3)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Success - Backend: {result.get('backend')}, Type: {search_type}")
            print(f"  Found {len(result.get('results', []))} results:")
            for i, res in enumerate(result.get('results', [])[:2], 1):
                print(f"    {i}. {res.get('title', 'N/A')}")


def test_brave_options():
    """Test Brave with different options."""
    print("\n" + "="*80)
    print("TESTING BRAVE - DIFFERENT OPTIONS")
    print("="*80)

    mock_interpreter = Mock()
    computer = Computer(mock_interpreter)

    # Test cases: (query, options, description)
    test_cases = [
        ("latest AI research breakthroughs", {"count": 5}, "Limit to 5 results - Fewer results"),
        ("python programming tutorials", {"safesearch": "strict"}, "Strict safe search - Filtered content"),
        ("Brexit impact on UK economy", {"country_code": "GB", "language_code": "en"}, "UK localized search - British results"),
    ]

    for query, options, description in test_cases:
        print(f"\n{'-'*80}")
        print(f"Test: {description}")
        print(f"Query: '{query}', Options: {options}")
        print(f"{'-'*80}")

        result = computer.search.search(query, backend="brave", **options)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Success - Backend: {result.get('backend')}")
            print(f"  Found {len(result.get('results', []))} results:")
            for i, res in enumerate(result.get('results', [])[:2], 1):
                print(f"    {i}. {res.get('title', 'N/A')}")


def test_tavily_options():
    """Test Tavily with different options."""
    print("\n" + "="*80)
    print("TESTING TAVILY - DIFFERENT OPTIONS")
    print("="*80)

    mock_interpreter = Mock()
    computer = Computer(mock_interpreter)

    # Test cases: (query, options, description)
    test_cases = [
        ("renewable energy trends 2024", {"search_depth": "basic", "max_results": 5}, "Basic search depth - Quick results"),
        ("machine learning interpretability methods", {"search_depth": "advanced"}, "Advanced search depth - Deeper analysis"),
        ("climate change impact studies", {"include_domains": ["arxiv.org", "nature.com", "science.org"]}, "Academic domains only - Research papers"),
    ]

    for query, options, description in test_cases:
        print(f"\n{'-'*80}")
        print(f"Test: {description}")
        print(f"Query: '{query}', Options: {options}")
        print(f"{'-'*80}")

        result = computer.search.search(query, backend="tavily", **options)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Success - Backend: {result.get('backend')}")
            print(f"  Found {len(result.get('results', []))} results:")
            for i, res in enumerate(result.get('results', [])[:2], 1):
                print(f"    {i}. {res.get('title', 'N/A')}")


def test_linkup_options():
    """Test LinkUp with different depth options."""
    print("\n" + "="*80)
    print("TESTING LINKUP - DIFFERENT OPTIONS")
    print("="*80)

    mock_interpreter = Mock()
    computer = Computer(mock_interpreter)

    # Test cases: (query, options, description)
    test_cases = [
        ("Python async programming guide", {"depth": "standard"}, "Standard depth - Balanced search"),
        ("neural network architecture optimization techniques", {"depth": "deep"}, "Deep search - Comprehensive results"),
    ]

    for query, options, description in test_cases:
        print(f"\n{'-'*80}")
        print(f"Test: {description}")
        print(f"Query: '{query}', Options: {options}")
        print(f"{'-'*80}")

        result = computer.search.search(query, backend="linkup", **options)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Success - Backend: {result.get('backend')}")
            print(f"  Found {len(result.get('results', []))} results:")
            for i, res in enumerate(result.get('results', [])[:2], 1):
                print(f"    {i}. {res.get('title', 'N/A')}")


def print_summary():
    """Print test summary and usage examples."""
    print("\n" + "="*80)
    print("TEST SUMMARY AND USAGE EXAMPLES")
    print("="*80)
    print("""
The unified search() method supports many backends and engines:

1. SERPAPI - 80+ search engines:
   - Google Scholar: engine="google_scholar"
   - YouTube: engine="youtube"
   - Bing, Yahoo, DuckDuckGo: engine="bing", "yahoo", "duckduckgo"
   - Shopping: engine="google_shopping", "ebay", "walmart"
   - Academic: engine="google_scholar"
   - News: engine="google_news"

2. SERPER - Google service types:
   - Scholar: type="scholar"
   - News: type="news"
   - Videos: type="videos"
   - Shopping: type="shopping"
   - Images: type="images"

3. BRAVE - Configurable options:
   - count, country, search_lang, safesearch, freshness

4. TAVILY - Advanced search:
   - search_depth ("basic" or "advanced")
   - include_domains, exclude_domains
   - include_raw_content, include_images

5. LINKUP - Depth control:
   - depth ("standard" or "deep")

EXAMPLE CODE:
    # Search Google Scholar for academic papers
    results = computer.search.search(
        "quantum computing",
        backend="serpapi",
        engine="google_scholar"
    )

    # Search YouTube for videos
    results = computer.search.search(
        "python tutorial",
        backend="serpapi",
        engine="youtube"
    )

    # Search Google News
    results = computer.search.search(
        "AI breakthroughs",
        backend="serper",
        type="news"
    )
""")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE SEARCH ENGINE TEST SUITE")
    print("="*80)
    print("\nThis will test various search engines and options across all backends.")
    print("Tests will skip backends where API keys are not set.")
    print("\nStarting tests...\n")

    try:
        test_serpapi_engines()
    except Exception as e:
        print(f"\n⚠️  SerpApi tests skipped or failed: {e}")

    try:
        test_serper_types()
    except Exception as e:
        print(f"\n⚠️  Serper tests skipped or failed: {e}")

    try:
        test_brave_options()
    except Exception as e:
        print(f"\n⚠️  Brave tests skipped or failed: {e}")

    try:
        test_tavily_options()
    except Exception as e:
        print(f"\n⚠️  Tavily tests skipped or failed: {e}")

    try:
        test_linkup_options()
    except Exception as e:
        print(f"\n⚠️  LinkUp tests skipped or failed: {e}")

    print_summary()

    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
