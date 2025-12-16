"""
Web search utilities.

This module provides web search related tools, with unified frontend methods
for search, fetch, answer, crawl, and structured output operations with
multiple backends.

Supported backends:
- Search: brave, tavily, linkup, serpapi, serper
- Answer: tavily, linkup
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

    def _search_brave(self, query, count=10, country_code=None, language_code=None, safesearch="moderate", **kwargs):
        """
        Search using Brave Search API backend.

        Args:
            query (str): The search query
            count (int): Number of results to return (default: 10, max: 20)
            country_code (str): 2-letter country code for localized results (e.g., "US", "GB", default: system locale)
            language_code (str): 2-letter language code for search (e.g., "en", "es", default: system locale)
            safesearch (str): Safe search level: "off", "moderate", or "strict" (default: "moderate")
            **kwargs: Additional Brave-specific parameters (freshness, text_decorations, spellcheck, etc.)

        Returns:
            Normalized dict with "results" key, or error dict
        """
        # Use locale-based defaults if not specified
        if country_code is None:
            country_code = self._default_country
        if language_code is None:
            language_code = self._default_lang

        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            return {
                "error": "BRAVE_API_KEY environment variable not set",
                "message": "To use Brave Search API, set the BRAVE_API_KEY environment variable. Get your API key at https://brave.com/search/api/",
                "alternative": "Try using a different backend (serper)"
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
            "country": country_code,
            "search_lang": language_code,
            "safesearch": safesearch,
            **kwargs
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Brave Search API request failed: {str(e)}",
                "message": "The API request encountered an error. Check your API key and internet connection.",
                "alternative": "Try using a different backend (serper)"
            }

        # Normalize Brave response format
        # Brave returns: {"web": {"results": [...]}, "news": {...}, ...}
        normalized = {
            "results": [],
            "raw_response": data
        }

        # Extract web results
        web_results = data.get("web", {}).get("results", [])
        for result in web_results:
            normalized_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("description", "")
            }
            normalized["results"].append(normalized_result)

        return normalized

    def _search_serper(self, query, num=10, type="search", country_code=None, language_code=None, autocorrect=True, **kwargs):
        """
        Search using Serper API (Google search) backend.

        Supports multiple search types: search, images, videos, news, shopping, places, maps, patents, autocomplete.

        Args:
            query (str): The search query
            num (int): Number of results to return (default: 10)
            type (str): Search type (default: "search")
                       Supported: "search", "images", "videos", "news", "shopping", "places", "maps", "patents", "autocomplete"
                       NOTE: "scholar" is NOT supported by Serper - use serpapi with engine="google_scholar" instead
            country_code (str): 2-letter country code for localized results (e.g., "us", "gb", default: system locale)
            language_code (str): 2-letter language code for interface (e.g., "en", "es", default: system locale)
            autocorrect (bool): Whether to autocorrect the query (default: True)
            **kwargs: Additional Serper-specific parameters (location, page, tbs, etc.)

        Returns:
            Normalized dict with "results" key, or error dict
        """
        # Validate search type
        supported_types = ["search", "images", "videos", "news", "shopping", "places", "maps", "patents", "autocomplete"]
        if type not in supported_types:
            return {
                "error": f"Unsupported search type: {type}",
                "message": f"Serper backend supports: {', '.join(supported_types)}",
                "alternative": f"For Google Scholar, use backend='serpapi' with engine='google_scholar'"
            }

        # Use locale-based defaults if not specified
        if country_code is None:
            country_code = self._default_country.lower()
        if language_code is None:
            language_code = self._default_lang

        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return {
                "error": "SERPER_API_KEY environment variable not set",
                "message": "To use Serper API, set the SERPER_API_KEY environment variable. Get your API key at https://serper.dev/",
                "alternative": "Try using a different backend (brave)"
            }

        # Map type to correct endpoint
        url = f"https://google.serper.dev/{type}"

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "q": query,
            "num": num,
            "gl": country_code,
            "hl": language_code,
            "autocorrect": autocorrect,
            **kwargs
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Serper API request failed: {str(e)}",
                "message": "The API request encountered an error. Check your API key and internet connection.",
                "alternative": "Try using a different backend (brave)"
            }

        # Normalize Serper response format
        # Response format varies by search type
        normalized = {
            "results": [],
            "raw_response": data
        }

        # Extract results based on search type - each type has its own response structure
        if type == "search":
            results_key = "organic"
        elif type == "news":
            results_key = "news"
        elif type == "images":
            results_key = "images"
        elif type == "videos":
            results_key = "videos"
        elif type == "shopping":
            results_key = "shopping"
        elif type == "places":
            results_key = "places"
        elif type == "maps":
            results_key = "places"  # Maps returns places
        elif type == "patents":
            results_key = "organic"  # Patents returns organic-style results
        elif type == "autocomplete":
            results_key = "suggestions"  # Autocomplete returns suggestions
        else:
            # Should not reach here due to validation above, but fallback
            results_key = "organic"

        results = data.get(results_key, [])
        for result in results:
            normalized_result = {
                "title": result.get("title", ""),
                "url": result.get("link", "") or result.get("url", ""),
                "snippet": result.get("snippet", "") or result.get("description", "")
            }
            normalized["results"].append(normalized_result)

        return normalized

    def _search_serpapi(self, query, num=10, engine="google", country_code=None, language_code=None, **kwargs):
        """
        Search using SerpApi backend (supports multiple search engines).

        Uses appropriate SerpApi classes for each engine:
        - Google-based engines (google, google_scholar, google_news, google_shopping, google_images, youtube):
          Uses GoogleSearch with engine parameter
        - Other engines: Uses specific classes (BingSearch, YahooSearch, etc.)

        Args:
            query (str): The search query
            num (int): Number of results to return (default: 10)
            engine (str): Search engine to use (default: "google")
                         Options: "google", "google_scholar", "google_news", "google_shopping",
                                  "google_images", "youtube", "bing", "yahoo", "duckduckgo",
                                  "baidu", "yandex", "ebay", "walmart", "home_depot",
                                  "apple_app_store", "naver", etc.
            country_code (str): 2-letter country code for localized results (e.g., "us", "gb", default: system locale)
            language_code (str): 2-letter language code for interface (e.g., "en", "es", default: system locale)
            **kwargs: Additional SerpApi parameters (location, google_domain, safe, start, filter, tbm, etc.)

        Returns:
            Normalized dict with "results" key, or error dict
        """
        # Use locale-based defaults if not specified
        if country_code is None:
            country_code = self._default_country.lower()
        if language_code is None:
            language_code = self._default_lang

        # Import SerpApi - try to get specific classes, fallback to GoogleSearch
        try:
            from serpapi import GoogleSearch
        except ImportError:
            return {
                "error": "google-search-results not installed",
                "message": "Install google-search-results: pip install google-search-results",
                "alternative": "Try using a different backend (brave, tavily)"
            }

        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return {
                "error": "SERPAPI_API_KEY environment variable not set",
                "message": "To use SerpApi, set the SERPAPI_API_KEY environment variable. Get your API key at https://serpapi.com/",
                "alternative": "Try using a different backend (brave, tavily)"
            }

        # Google-based engines use GoogleSearch with engine parameter
        # These work with GoogleSearch class + engine parameter
        # NOTE: YouTube should use YoutubeSearch class, not GoogleSearch with engine="youtube"
        google_engines = ["google", "google_scholar", "google_news", "google_shopping", "google_images"]

        # Engines that should use specific classes (if available)
        specific_class_engines = {
            "youtube": "YoutubeSearch",  # YouTube should use YoutubeSearch class
            "bing": "BingSearch",
            "yahoo": "YahooSearch",
            "duckduckgo": "DuckDuckGoSearch",
            "baidu": "BaiduSearch",
            "yandex": "YandexSearch",
            "ebay": "EbaySearch",
            "walmart": "WalmartSearch",
            "home_depot": "HomeDepotSearch",
            "apple_app_store": "AppleAppStoreSearch",
            "naver": "NaverSearch",
        }

        try:
            if engine in google_engines:
                # Use GoogleSearch with engine parameter for Google-based engines
                search_params = {
                    "q": query,
                    "num": num,
                    "engine": engine,
                    "api_key": api_key,
                    "gl": country_code,
                    "hl": language_code,
                    **kwargs
                }
                search = GoogleSearch(search_params)
                data = search.get_dict()
            elif engine in specific_class_engines:
                # Use specific class for non-Google engines
                class_name = specific_class_engines[engine]
                try:
                    # Dynamically import the specific class
                    module = __import__("serpapi", fromlist=[class_name])
                    SearchClass = getattr(module, class_name)
                except AttributeError:
                    return {
                        "error": f"SerpApi class {class_name} not found",
                        "message": f"The {class_name} class is not available in your serpapi package version. Update google-search-results: pip install --upgrade google-search-results",
                        "alternative": "Try using a different engine or backend"
                    }

                # Build params - different engines use different query parameter names
                search_params = {"api_key": api_key}
                if engine == "yahoo":
                    search_params["p"] = query  # Yahoo uses "p" instead of "q"
                elif engine == "ebay":
                    search_params["_nkw"] = query  # eBay uses "_nkw"
                elif engine == "youtube":
                    search_params["q"] = query  # YouTube uses "q" (per SerpApi quick start docs)
                    if num:
                        search_params["num"] = num
                else:
                    search_params["q"] = query
                    if num:
                        search_params["num"] = num

                # Add localization for engines that support it
                if engine in ["bing", "yahoo", "duckduckgo", "youtube"]:
                    search_params["gl"] = country_code
                    search_params["hl"] = language_code

                search_params.update(kwargs)
                search = SearchClass(search_params)
                data = search.get_dict()
            else:
                # Unknown engine - reject it
                return {
                    "error": f"Unknown SerpApi engine: {engine}",
                    "message": f"Engine '{engine}' is not supported. Supported engines: {', '.join(google_engines + list(specific_class_engines.keys()))}",
                    "alternative": "Use a supported engine or try a different backend"
                }
        except Exception as e:
            return {
                "error": f"SerpApi request failed: {str(e)}",
                "message": "The API request encountered an error. Check your API key and internet connection.",
                "alternative": "Try using a different backend (brave, tavily)"
            }

        # Normalize SerpApi response format
        # Different engines return results in different keys
        normalized = {
            "results": [],
            "raw_response": data
        }

        # Map engine to response key
        # Different engines return results in different keys
        engine_result_keys = {
            "google": "organic_results",
            "google_scholar": "organic_results",
            "bing": "organic_results",
            "yahoo": "organic_results",
            "duckduckgo": "organic_results",
            "youtube": "video_results",  # YouTube returns video_results
            "google_news": "news_results",
            "google_shopping": "shopping_results",
            "google_images": "images_results",
            "ebay": "organic_results",  # eBay returns organic_results
            "walmart": "organic_results",
            "home_depot": "organic_results",
            "apple_app_store": "organic_results",
            "naver": "organic_results",
            "baidu": "organic_results",
            "yandex": "organic_results",
        }

        # Get the appropriate results key for this engine
        results_key = engine_result_keys.get(engine, "organic_results")
        results = data.get(results_key, [])

        # If no results found in the expected key, fail loudly with debug info
        if not results:
            available_keys = [k for k in data.keys() if isinstance(data.get(k), list)]
            return {
                "error": f"No results found in expected key '{results_key}' for engine '{engine}'",
                "message": f"SerpApi response did not contain results in the expected key. Available list keys: {available_keys}",
                "raw_response": data,
                "alternative": "Check the SerpApi response structure or try a different engine"
            }

        # Extract and normalize results
        for result in results:
            # Different engines use different field names
            title = result.get("title", "") or result.get("name", "") or result.get("product_title", "")
            url = result.get("link", "") or result.get("url", "") or result.get("href", "")
            snippet = result.get("snippet", "") or result.get("description", "") or result.get("about_this_result", {}).get("source", {}).get("description", "")

            # For YouTube videos, include video info
            if engine == "youtube" and "link" in result:
                url = result.get("link", "")
                snippet = result.get("description", "") or f"Video by {result.get('channel', {}).get('name', 'Unknown')}"

            # For shopping, include price if available
            if engine == "google_shopping" and "price" in result:
                price = result.get("price", "")
                if price:
                    snippet = f"{snippet} - {price}".strip()

            normalized_result = {
                "title": title,
                "url": url,
                "snippet": snippet
            }
            normalized["results"].append(normalized_result)

        return normalized

    def _search_tavily(self, query, max_results=10, country_code=None, language_code=None, **kwargs):
        """
        Search using Tavily backend (just search results, no AI answer).

        Args:
            query (str): The search query
            max_results (int): Maximum number of results to return (default: 10)
            country_code (str): 2-letter country code (not directly used by Tavily, but kept for consistency)
            language_code (str): 2-letter language code (not directly used by Tavily, but kept for consistency)
            **kwargs: Additional Tavily search parameters (search_depth, include_domains, exclude_domains, etc.)

        Returns:
            Normalized dict with "results" key, or error dict
        """
        # Tavily doesn't use country/language codes directly, but we accept them for consistency
        try:
            from tavily import TavilyClient
        except ImportError:
            return {
                "error": "tavily-python not installed",
                "message": "Install tavily-python: pip install tavily-python",
                "alternative": "Try using a different backend (brave, linkup)"
            }

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {
                "error": "TAVILY_API_KEY environment variable not set",
                "message": "To use Tavily, set the TAVILY_API_KEY environment variable.",
                "alternative": "Try using a different backend (brave, linkup)"
            }

        try:
            client = TavilyClient(api_key=api_key)

            # Build search parameters - explicitly exclude AI answer
            search_params = {
                "query": query,
                "max_results": max_results,
                "include_answer": False,  # Just search results, no AI answer
                **kwargs
            }

            response = client.search(**search_params)
        except Exception as e:
            return {
                "error": f"Tavily API request failed: {str(e)}",
                "message": "The Tavily API request encountered an error.",
                "alternative": "Try using a different backend (brave, linkup)"
            }

        # Normalize Tavily response format
        # Tavily returns: {"results": [{"title": str, "url": str, "content": str, ...}]}
        normalized = {
            "results": [],
            "raw_response": response
        }

        # Extract results
        results = response.get("results", [])
        for result in results:
            normalized_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", "")[:200] if result.get("content") else ""
            }
            normalized["results"].append(normalized_result)

        return normalized

    def _search_linkup(self, query, depth="standard", country_code=None, language_code=None, **kwargs):
        """
        Search using LinkUp backend (searchResults mode).

        Args:
            query (str): The search query
            depth (str): "standard" or "deep" (default: "standard")
            country_code (str): 2-letter country code (not directly used by LinkUp, but kept for consistency)
            language_code (str): 2-letter language code (not directly used by LinkUp, but kept for consistency)
            **kwargs: Additional LinkUp search parameters

        Returns:
            Normalized dict with "results" key, or error dict
        """
        # LinkUp doesn't use country/language codes directly, but we accept them for consistency
        try:
            from linkup import LinkupClient
        except ImportError:
            return {
                "error": "linkup-sdk not installed",
                "message": "Install linkup-sdk: pip install linkup-sdk",
                "alternative": "Try using a different backend (brave, tavily)"
            }

        api_key = os.getenv("LINKUP_API_KEY")
        if not api_key:
            return {
                "error": "LINKUP_API_KEY environment variable not set",
                "message": "To use LinkUp, set the LINKUP_API_KEY environment variable.",
                "alternative": "Try using a different backend (brave, tavily)"
            }

        try:
            client = LinkupClient(api_key=api_key)

            # Build search parameters
            search_params = {
                "query": query,
                "depth": depth,
                "output_type": "searchResults",  # Just search results, no AI answer
                **kwargs
            }

            response = client.search(**search_params)
        except Exception as e:
            return {
                "error": f"LinkUp API request failed: {str(e)}",
                "message": "The LinkUp API request encountered an error.",
                "alternative": "Try using a different backend (brave, tavily)"
            }

        # Normalize LinkUp response format
        # LinkUp returns a LinkupSearchResults object with .results attribute
        normalized = {
            "results": [],
            "raw_response": response
        }

        # Extract results - check if it's a list of objects or dicts
        results = getattr(response, "results", [])
        if not isinstance(results, list):
            raise ValueError(
                f"LinkUp response 'results' is not a list: {type(results).__name__}. "
                f"Response type: {type(response).__name__}"
            )

        for result in results:
            # Handle both object attributes and dict keys
            if hasattr(result, "name"):
                # Object with attributes
                normalized_result = {
                    "title": getattr(result, "name", ""),
                    "url": getattr(result, "url", ""),
                    "snippet": getattr(result, "content", "")[:200] if getattr(result, "content", None) else ""
                }
            elif isinstance(result, dict):
                # Dict
                normalized_result = {
                    "title": result.get("name", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", "")[:200] if result.get("content") else ""
                }
            else:
                raise ValueError(
                    f"LinkUp result item is neither object nor dict: {type(result).__name__}. "
                    f"Result: {str(result)[:200]}"
                )
            normalized["results"].append(normalized_result)

        return normalized

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

    def search(self, query: str, backend: Optional[str] = None, country_code: Optional[str] = None, language_code: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Search the web for information.

        This method automatically selects the best available backend or uses
        the specified one. Backends are tried in order: brave, tavily, linkup, serpapi, serper.

        Args:
            query (str): The search query
            backend (str, optional): Force a specific backend ("brave", "tavily", "linkup", "serpapi", or "serper").
                                     If None, auto-selects based on availability.
            country_code (str, optional): 2-letter country code for localized results (e.g., "US", "GB", "FR").
                                          Defaults to system locale. Supported by: brave, serpapi, serper.
            language_code (str, optional): 2-letter language code for search interface (e.g., "en", "es", "fr").
                                           Defaults to system locale. Supported by: brave, serpapi, serper.
            **kwargs: Additional backend-specific parameters:

                BRAVE (2000 free/month):
                    - count (int, 1-20): Number of results (default: 10, max: 20)
                    - safesearch (str): "off", "moderate", or "strict" (default: "moderate")
                    - freshness (str): "pd" (past day), "pw" (past week), "pm" (past month), "py" (past year)
                    - text_decorations (bool): Include text decorations in snippets (default: True)
                    - spellcheck (bool): Enable spellcheck (default: True)
                    NOTE: Use country_code and language_code parameters (not country/search_lang)

                TAVILY (1000 free/month):
                    - max_results (int): Number of results to return (default: 10)
                    - search_depth (str): "basic" or "advanced" (default: "basic")
                    - include_domains (list): List of domains to include (e.g., ["example.com"])
                    - exclude_domains (list): List of domains to exclude
                    - include_raw_content (bool): Include full HTML content (default: False)
                    - include_images (bool): Include images in results (default: False)
                    - topic (str): "general" or "news" (default: "general")
                    - days (int): Number of days back to search (for topic="news")

                LINKUP (1000 free/month):
                    - depth (str): "standard" or "deep" (default: "standard")
                    - output_type (str): "searchResults" (default), "sourcedAnswer", or "structured"
                    - structured_output_schema (dict): Schema for structured output mode

                SERPAPI (250 free/month) - supports 80+ search engines:
                    - num (int): Number of results (default: 10)
                    - engine (str): Search engine to use (default: "google")
                        Common engines:
                          - "google" (default): Regular Google search
                          - "google_scholar": Academic papers, citations
                          - "bing": Microsoft Bing search
                          - "yahoo": Yahoo search
                          - "duckduckgo": DuckDuckGo search
                          - "baidu": Chinese search engine
                          - "yandex": Russian search engine
                          - "youtube": YouTube video search
                          - "google_news": Google News search
                          - "google_images": Google Images search
                          - "google_shopping": Google Shopping search
                          - "ebay": eBay product search
                          - "walmart": Walmart product search
                          - "home_depot": Home Depot search
                          - "apple_app_store": App Store search
                          - "google_play": Google Play Store search
                    - location (str): Location for localized results (e.g., "Austin, Texas")
                    - google_domain (str): Google domain (e.g., "google.com", "google.co.uk")
                    - safe (str): Safe search - "active" or "off"
                    - start (int): Pagination offset
                    - filter (str): Duplicate filter - "0" (off) or "1" (on)
                    - tbm (str): Search type - "nws" (news), "isch" (images), "vid" (videos), "shop" (shopping)
                    NOTE: Use country_code and language_code parameters (not gl/hl). Each engine has specific parameters. See https://serpapi.com/ for details.

                SERPER (2500 total, non-renewable) - supports multiple search types:
                    - num (int): Number of results (default: 10)
                    - type (str): Search type (default: "search")
                        Supported types:
                          - "search": Regular web search (default)
                          - "images": Image search
                          - "videos": Video search
                          - "places": Google Maps places search
                          - "maps": Google Maps search
                          - "news": News search
                          - "shopping": Shopping search
                          - "patents": Patent search
                          - "autocomplete": Search suggestions
                        NOTE: "scholar" is NOT supported by Serper. Use serpapi with engine="google_scholar" instead.
                    - location (str): Location for localized results
                    - autocorrect (bool): Enable query autocorrection (default: True)
                    - page (int): Page number for pagination
                    - tbs (str): Time-based search (e.g., "qdr:d" for past day, "qdr:w" for past week)
                    NOTE: Use country_code and language_code parameters (not gl/hl)

        Returns:
            dict: Normalized response with:
                - "results" (list): List of result dicts with "title", "url", "snippet"
                - "raw_response" (dict): Original backend response (structure varies by backend)
                - "backend" (str): Backend that was used
                OR error dict with "error", "message", "alternative" keys

        Examples:
            # Basic search (auto-selects backend)
            results = computer.search.search("machine learning tutorials")
            for result in results["results"]:
                print(f"{result['title']}: {result['url']}")

            # Search Google Scholar for academic papers (using serpapi)
            results = computer.search.search(
                "quantum computing",
                backend="serpapi",
                engine="google_scholar"
            )

            # Search YouTube videos (using serpapi)
            results = computer.search.search(
                "python tutorial",
                backend="serpapi",
                engine="youtube"
            )

            # Search Google News (using serper)
            results = computer.search.search(
                "AI breakthroughs",
                backend="serper",
                type="news"
            )

            # Deep web search with specific domains (using tavily)
            results = computer.search.search(
                "climate change research",
                backend="tavily",
                search_depth="advanced",
                include_domains=["nature.com", "science.org"]
            )

            # Shopping search (using serpapi)
            results = computer.search.search(
                "laptop",
                backend="serpapi",
                engine="google_shopping"
            )
        """
        used_backend = None

        # Prepare normalized parameters for all backends
        backend_kwargs = kwargs.copy()
        backend_kwargs['country_code'] = country_code
        backend_kwargs['language_code'] = language_code

        if backend:
            backend = backend.lower()
            backend_methods = {
                "brave": self._search_brave,
                "tavily": self._search_tavily,
                "linkup": self._search_linkup,
                "serpapi": self._search_serpapi,
                "serper": self._search_serper
            }

            if backend not in backend_methods:
                error_result = {
                    "error": f"Unknown backend: {backend}",
                    "message": f"Supported backends for search: {', '.join(backend_methods.keys())}",
                    "alternative": "Try without specifying a backend to auto-select"
                }
                print(f"❌ Error: {error_result['error']}")
                print(f"   {error_result['message']}")
                return error_result

            result = backend_methods[backend](query, **backend_kwargs)
            if "error" not in result:
                used_backend = backend
                result["backend"] = used_backend
                self._print_search_result(query, result, used_backend)
                return result

            # If specified backend failed, return the error
            print(f"❌ {backend} backend failed: {result.get('error', 'Unknown error')}")
            if result.get('alternative'):
                print(f"   {result['alternative']}")
            return result

        # Auto-select backend
        # Priority order: brave (2k/month) > tavily (1k/month) > linkup (1k/month) > serpapi (250/month) > serper (2.5k total)
        backends_to_try = ["brave", "tavily", "linkup", "serpapi", "serper"]

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue

            if backend_name == "brave":
                result = self._search_brave(query, **backend_kwargs)
            elif backend_name == "tavily":
                result = self._search_tavily(query, **backend_kwargs)
            elif backend_name == "linkup":
                result = self._search_linkup(query, **backend_kwargs)
            elif backend_name == "serpapi":
                result = self._search_serpapi(query, **backend_kwargs)
            elif backend_name == "serper":
                result = self._search_serper(query, **backend_kwargs)
            else:
                continue

            if "error" not in result:
                used_backend = backend_name
                result["backend"] = used_backend
                self._print_search_result(query, result, used_backend)
                return result

        # All backends failed or unavailable
        error_result = {
            "error": "No available backends",
            "message": "No search backends are available. Set one of: BRAVE_API_KEY, TAVILY_API_KEY, LINKUP_API_KEY, SERPAPI_API_KEY, or SERPER_API_KEY.",
            "alternative": "Get API keys at: https://brave.com/search/api/, https://tavily.com/, https://linkup.so/, https://serpapi.com/, or https://serper.dev/"
        }
        print(f"❌ {error_result['error']}")
        print(f"   {error_result['message']}")
        return error_result

    def _answer_tavily(self, question, answer_mode="basic", **kwargs):
        """
        Get AI-generated answer using Tavily backend.

        Args:
            question: The question to answer
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
                "query": question,
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

    def _answer_linkup(self, question, depth="standard", **kwargs):
        """
        Get AI-generated answer using LinkUp backend.

        Args:
            question: The question to answer
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
                "query": question,
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

        # LinkUp returns a LinkupSourcedAnswer object, not a dict
        # Access attributes directly: response.answer, response.sources
        normalized = {
            "answer": getattr(response, "answer", ""),
            "sources": []
        }

        # Extract sources - check if it's a list of objects or dicts
        sources = getattr(response, "sources", [])
        if not isinstance(sources, list):
            raise ValueError(
                f"LinkUp response 'sources' is not a list: {type(sources).__name__}. "
                f"Response type: {type(response).__name__}, "
                f"Response attributes: {dir(response)}"
            )

        for source in sources:
            # Handle both object attributes and dict keys
            if hasattr(source, "name"):
                # Object with attributes
                normalized_source = {
                    "title": getattr(source, "name", ""),
                    "url": getattr(source, "url", ""),
                    "snippet": getattr(source, "snippet", "")[:200] if getattr(source, "snippet", None) else ""
                }
            elif isinstance(source, dict):
                # Dict
                normalized_source = {
                    "title": source.get("name", ""),
                    "url": source.get("url", ""),
                    "snippet": source.get("snippet", "")[:200] if source.get("snippet") else ""
                }
            else:
                raise ValueError(
                    f"LinkUp source item is neither object nor dict: {type(source).__name__}. "
                    f"Source: {str(source)[:200]}"
                )
            normalized["sources"].append(normalized_source)

        return normalized

    def _print_search_result(self, query: str, result: Dict[str, Any], backend: str):
        """Print formatted search result to help the AI understand what was found."""
        results = result.get("results", [])

        print(f"\n🔍 Search results for '{query}' (using `{backend}` backend):")
        print(f"Found {len(results)} results\n")

        if results:
            print("Top results:")
            for i, res in enumerate(results[:5], 1):  # Show first 5 results
                title = res.get("title", "No title")
                url = res.get("url", "")
                print(f"  {i}. {title}")
                if url:
                    print(f"     {url}")
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more results")
            print()

    def _print_answer_result(self, question: str, result: Dict[str, Any], backend: str):
        """Print formatted answer result to help the AI understand what was found."""
        answer_text = result.get("answer", "")
        sources = result.get("sources", [])

        # Backend library information
        backend_info = {
            "tavily": ("tavily", "TavilyClient", "from tavily import TavilyClient"),
            "linkup": ("linkup-sdk", "LinkupClient", "from linkup import LinkupClient"),
        }
        lib_name, client_name, import_stmt = backend_info.get(backend, ("", "", ""))

        print(f"\n📝 Answer (using `{backend}` backend):")
        print(f"{answer_text}\n")

        if sources:
            print(f"📚 Sources ({len(sources)}):")
            for i, source in enumerate(sources[:3], 1):  # Show first 3 sources
                title = source.get("title", "No title")
                print(f"  {i}. {title}")
            if len(sources) > 3:
                print(f"  ... and {len(sources) - 3} more sources")
            print()

        if lib_name:
            print(f"💡 For more control, use {lib_name} directly: {import_stmt}")
            print()

    def answer(self, question: str, backend: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get AI-generated answer with web sources. PREFERRED for questions requiring a direct answer about current web knowledge.

        This method automatically selects the best available backend or uses
        the specified one. Backends are tried in order: linkup, tavily.

        Args:
            question (str): The question to answer
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
        used_backend = None

        if backend:
            backend = backend.lower()
            if backend == "tavily":
                result = self._answer_tavily(question, **kwargs)
                if "error" not in result:
                    used_backend = "tavily"
                    result["backend"] = used_backend
                    self._print_answer_result(question, result, used_backend)
                    return result
            elif backend == "linkup":
                result = self._answer_linkup(question, **kwargs)
                if "error" not in result:
                    used_backend = "linkup"
                    result["backend"] = used_backend
                    self._print_answer_result(question, result, used_backend)
                    return result
            else:
                error_result = {
                    "error": f"Unknown backend: {backend}",
                    "message": f"Supported backends for answer: 'tavily', 'linkup'",
                    "alternative": "Try without specifying a backend to auto-select"
                }
                print(f"❌ Error: {error_result['error']}")
                print(f"   {error_result['message']}")
                return error_result
            # If specified backend failed, return the error
            print(f"❌ {backend} backend failed: {result.get('error', 'Unknown error')}")
            if result.get('alternative'):
                print(f"   {result['alternative']}")
            return result

        # Auto-select backend: prefer linkup over tavily
        backends_to_try = ["linkup", "tavily"]

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue

            if backend_name == "linkup":
                result = self._answer_linkup(question, **kwargs)
            elif backend_name == "tavily":
                result = self._answer_tavily(question, **kwargs)
            else:
                continue

            if "error" not in result:
                used_backend = backend_name
                result["backend"] = used_backend
                self._print_answer_result(question, result, used_backend)
                return result

        # All backends failed or unavailable
        error_result = {
            "error": "No available backends",
            "message": "No answer backends are available. Set TAVILY_API_KEY or LINKUP_API_KEY environment variable.",
            "alternative": "Install required packages: pip install tavily-python linkup-sdk"
        }
        print(f"❌ {error_result['error']}")
        print(f"   {error_result['message']}")
        return error_result
