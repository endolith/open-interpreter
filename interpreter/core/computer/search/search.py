"""
Web search utilities.

This module provides web search related tools, with unified frontend methods
for search, fetch, answer, crawl, and structured output operations with
multiple backends.
"""

import os
import json
import locale
import requests
from typing import Optional, Dict, List, Any


class Search:
    def __init__(self, computer):
        self.computer = computer
        loc = locale.getdefaultlocale()[0]
        # Remove encoding suffix if present (e.g., 'en_US.UTF-8' -> 'en_US')
        loc = loc.split('.')[0]
        # Split on underscore to get language and country (standard format: 'en_US')
        lang, country = loc.split('_', 1)
        self._default_lang = lang.lower()
        self._default_country = country.upper()

    def brave(self, query, count=10, country=None, search_lang=None, safesearch="moderate"):
        """
        Search using Brave Search API.

        Args:
            query (str): The search query
            count (int): Number of results to return (default: 10, max: 20)
            country (str): Country code for localized results (default: system locale)
            search_lang (str): Language code for search (default: system locale)
            safesearch (str): Safe search level: "off", "moderate", or "strict" (default: "moderate")

        Returns:
            dict: {"web": {"results": [{"title": str, "url": str, "description": str, ...}], ...}, "news": {...}, ...}

        Example:
            results = computer.search.brave("artificial intelligence", count=5)
            for result in results.get("web", {}).get("results", []):
                print(result["title"], result["url"])
        """
        # Use locale-based defaults if not specified
        if country is None:
            country = self._default_country
        if search_lang is None:
            search_lang = self._default_lang

        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            return {
                "error": "BRAVE_API_KEY environment variable not set",
                "message": "To use Brave Search API, set the BRAVE_API_KEY environment variable. Get your API key at https://brave.com/search/api/",
                "alternative": "Try using a different search method instead"
            }

        url = "https://api.search.brave.com/res/v1/web/search"

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }

        params = {
            "q": query,
            "count": min(count, 20),  # Brave has a max of 20
            "country": country,
            "search_lang": search_lang,
            "safesearch": safesearch
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Brave Search API request failed: {str(e)}",
                "message": "The API request encountered an error. Check your API key and internet connection.",
                "alternative": "Try using a different search method instead"
            }

    def serper(self, query, num=10, gl=None, hl=None, autocorrect=True):
        """
        Search using Serper API (Google search).

        Args:
            query (str): The search query
            num (int): Number of results to return (default: 10)
            gl (str): Country code for localized results (default: system locale)
            hl (str): Language code for interface (default: system locale)
            autocorrect (bool): Whether to autocorrect the query (default: True)

        Returns:
            dict: {"searchParameters": {"q": str, "gl": str, "hl": str, "num": int, ...}, "organic": [{"title": str, "link": str, "snippet": str, "date": str, "sitelinks": [...]}, ...], "answerBox": {...}, "knowledgeGraph": {...}, ...}

        Example:
            results = computer.search.serper("machine learning tutorials")
            for result in results.get("organic", []):
                print(result["title"], result["link"])
        """
        # Use locale-based defaults if not specified
        if gl is None:
            gl = self._default_country.lower()
        if hl is None:
            hl = self._default_lang

        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return {
                "error": "SERPER_API_KEY environment variable not set",
                "message": "To use Serper API, set the SERPER_API_KEY environment variable. Get your API key at https://serper.dev/",
                "alternative": "Try using a different search method instead"
            }

        url = "https://google.serper.dev/search"

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "q": query,
            "num": num,
            "gl": gl,
            "hl": hl,
            "autocorrect": autocorrect
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Serper API request failed: {str(e)}",
                "message": "The API request encountered an error. Check your API key and internet connection.",
                "alternative": "Try using a different search method instead"
            }

    def _check_backend_available(self, backend: str) -> bool:
        """Check if a backend is available (has API key)."""
        backend_keys = {
            "tavily": "TAVILY_API_KEY",
            "linkup": "LINKUP_API_KEY",
            "serper": "SERPER_API_KEY",
            "brave": "BRAVE_API_KEY",
            "serpapi": "SERPAPI_API_KEY",
        }
        key_name = backend_keys.get(backend.lower())
        if not key_name:
            return False
        return bool(os.getenv(key_name))

    def _answer_tavily(self, query, answer_mode="basic", **kwargs):
        """
        Get AI-generated answer using Tavily backend.

        Args:
            query: The search query
            answer_mode: "basic" or "advanced" (default: "basic")
            **kwargs: Additional Tavily search parameters

        Returns:
            Normalized dict with "answer" and "sources" keys, or error dict
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            return {
                "error": "tavily-python not installed",
                "message": "Install tavily-python: pip install tavily-python",
                "alternative": "Try using a different backend (linkup)"
            }

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {
                "error": "TAVILY_API_KEY environment variable not set",
                "message":
                "To use Tavily, set the TAVILY_API_KEY environment variable.",
                "alternative": "Try using a different backend (linkup)"
            }

        try:
            client = TavilyClient(api_key=api_key)

            # Build search parameters
            search_params = {
                "query": query,
                "include_answer": answer_mode,  # "basic" or "advanced"
                **kwargs
            }

            response = client.search(**search_params)
        except Exception as e:
            # API request failed (network error, rate limit, etc.) - return error dict for fallback
            return {
                "error": f"Tavily API request failed: {str(e)}",
                "message": "The Tavily API request encountered an error.",
                "alternative": "Try using a different backend (linkup)"
            }

        # Response format validation - raise exception for programming errors
        # Tavily returns: {"answer": str, "results": [{"title": str, "url": str, "content": str, ...}, ...]}
        if not isinstance(response, dict):
            raise ValueError(
                f"Tavily returned unexpected response type: {type(response).__name__}. "
                f"Expected dict, got {type(response).__name__}. "
                f"Response: {str(response)[:500]}"
            )

        normalized = {
            "answer": response.get("answer", ""),
            "sources": []
        }

        # Extract sources from results
        results = response.get("results", [])
        if not isinstance(results, list):
            raise ValueError(
                f"Tavily response missing or invalid 'results' field. "
                f"Expected list, got {type(results).__name__}. "
                f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}"
            )

        for result in results:
            if not isinstance(result, dict):
                raise ValueError(
                    f"Tavily result item is not a dict: {type(result).__name__}. "
                    f"Result: {str(result)[:200]}"
                )
            source = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", "")[:200] if result.get("content") else ""
            }
            normalized["sources"].append(source)

        return normalized

    def _answer_linkup(self, query, depth="standard", **kwargs):
        """
        Get AI-generated answer using LinkUp backend.

        Args:
            query: The search query
            depth: "standard" or "deep" (default: "standard")
            **kwargs: Additional LinkUp search parameters

        Returns:
            Normalized dict with "answer" and "sources" keys, or error dict
        """
        try:
            from linkup import LinkupClient
        except ImportError:
            return {
                "error": "linkup-sdk not installed",
                "message": "Install linkup-sdk: pip install linkup-sdk",
                "alternative": "Try using a different backend (tavily)"
            }

        api_key = os.getenv("LINKUP_API_KEY")
        if not api_key:
            return {
                "error": "LINKUP_API_KEY environment variable not set",
                "message":
                "To use LinkUp, set the LINKUP_API_KEY environment variable.",
                "alternative": "Try using a different backend (tavily)"
            }

        try:
            client = LinkupClient(api_key=api_key)

            # Build search parameters
            search_params = {
                "query": query,
                "depth": depth,
                "output_type": "sourcedAnswer",
                **kwargs
            }

            response = client.search(**search_params)
        except Exception as e:
            # API request failed (network error, rate limit, etc.) - return error dict for fallback
            return {
                "error": f"LinkUp API request failed: {str(e)}",
                "message": "The LinkUp API request encountered an error.",
                "alternative": "Try using a different backend (tavily)"
            }

        # Response format validation - raise exception for programming errors
        # LinkUp sourcedAnswer returns: {"answer": str, "sources": [{"name": str, "url": str, "snippet": str}, ...]}
        if not isinstance(response, dict):
            raise ValueError(
                f"LinkUp returned unexpected response type: {type(response).__name__}. "
                f"Expected dict, got {type(response).__name__}. "
                f"Response: {str(response)[:500]}"
            )

        normalized = {
            "answer": response.get("answer", ""),
            "sources": []
        }

        # Extract sources
        sources = response.get("sources", [])
        if not isinstance(sources, list):
            raise ValueError(
                f"LinkUp response missing or invalid 'sources' field. "
                f"Expected list, got {type(sources).__name__}. "
                f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}"
            )

        for source in sources:
            if not isinstance(source, dict):
                raise ValueError(
                    f"LinkUp source item is not a dict: {type(source).__name__}. "
                    f"Source: {str(source)[:200]}"
                )
            normalized_source = {
                "title": source.get("name", ""),
                "url": source.get("url", ""),
                "snippet": source.get("snippet", "")[:200] if source.get("snippet") else ""
            }
            normalized["sources"].append(normalized_source)

        return normalized

    def answer(self, query: str, backend: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get an AI-generated answer to a query with source citations.

        This method automatically selects the best available backend or uses
        the specified one. Backends are tried in order: linkup, tavily.

        Args:
            query (str): The question or query to answer
            backend (str, optional): Force a specific backend ("tavily" or "linkup").
                                     If None, auto-selects based on availability.
            **kwargs: Additional backend-specific parameters:
                - For tavily: answer_mode ("basic" or "advanced"), search_depth, etc.
                - For linkup: depth ("standard" or "deep"), include_inline_citations, etc.

        Returns:
            dict: Normalized response with:
                - "answer" (str): The AI-generated answer
                - "sources" (list): List of source dicts with "title", "url", "snippet"
                OR error dict with "error", "message", "alternative" keys

        Example:
            result = computer.search.answer("What is machine learning?")
            print(result["answer"])
            for source in result["sources"]:
                print(f"- {source['title']}: {source['url']}")
        """
        if backend:
            backend = backend.lower()
            if backend == "tavily":
                result = self._answer_tavily(query, **kwargs)
                if "error" not in result:
                    return result
            elif backend == "linkup":
                result = self._answer_linkup(query, **kwargs)
                if "error" not in result:
                    return result
            else:
                return {
                    "error": f"Unknown backend: {backend}",
                    "message": f"Supported backends for answer: 'tavily', 'linkup'",
                    "alternative": "Try without specifying a backend to auto-select"
                }
            # If specified backend failed, return the error
            return result

        # Auto-select backend: prefer linkup over tavily
        backends_to_try = ["linkup", "tavily"]

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue

            if backend_name == "linkup":
                result = self._answer_linkup(query, **kwargs)
            elif backend_name == "tavily":
                result = self._answer_tavily(query, **kwargs)
            else:
                continue

            if "error" not in result:
                return result

        # All backends failed or unavailable
        return {
            "error": "No available backends",
            "message": "No answer backends are available. Set TAVILY_API_KEY or LINKUP_API_KEY environment variable.",
            "alternative": "Install required packages: pip install tavily-python linkup-sdk"
        }
