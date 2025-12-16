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


class ApiKeyError(Exception):
    """Exception raised when an API key is missing. Contains error dict."""
    def __init__(self, error_dict):
        self.error_dict = error_dict
        super().__init__(error_dict.get("error", "API key missing"))


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

    def _get_locale_defaults(self, country_code=None, language_code=None, country_case="lower"):
        """
        Get locale defaults for country_code and language_code.

        Args:
            country_code: Optional country code (if None, uses default)
            language_code: Optional language code (if None, uses default)
            country_case: "lower" or "upper" for country code case (default: "lower")

        Returns:
            tuple: (country_code, language_code) with defaults applied
        """
        if country_code is None:
            country_code = self._default_country.lower() if country_case == "lower" else self._default_country
        if language_code is None:
            language_code = self._default_lang
        return country_code, language_code

    def _check_api_key(self, key_name):
        """
        Check if an API key is set and return it.

        Args:
            key_name: Environment variable name (e.g., "BRAVE_API_KEY")

        Returns:
            str: API key if found

        Raises:
            ApiKeyError: If API key is missing (error_dict is in exception)
        """
        # Mapping of API key names to (backend_name, key_url)
        api_key_info = {
            "BRAVE_API_KEY": ("Brave Search API", "https://brave.com/search/api/"),
            "SERPER_API_KEY": ("Serper API", "https://serper.dev/"),
            "SERPAPI_API_KEY": ("SerpApi", "https://serpapi.com/"),
            "TAVILY_API_KEY": ("Tavily", "https://tavily.com/"),
            "LINKUP_API_KEY": ("LinkUp", "https://linkup.so/"),
        }

        api_key = os.getenv(key_name)
        if not api_key:
            backend_name, key_url = api_key_info.get(key_name, ("this backend", ""))
            url_text = f" Get your API key at {key_url}" if key_url else ""
            error_dict = {
                "error": f"{key_name} environment variable not set",
                "message": f"To use {backend_name}, set the {key_name} environment variable.{url_text}",
                "alternative": "Try using a different backend"
            }
            raise ApiKeyError(error_dict)
        return api_key

    def _handle_import_error(self, package_name, install_cmd):
        """Handle ImportError for a package."""
        return {
            "error": f"{package_name} not installed",
            "message": f"Install {package_name}: {install_cmd}",
            "alternative": "Try using a different backend"
        }

    def _handle_api_request_error(self, backend_name, error):
        """Create standardized error dict for API request failures."""
        return {
            "error": f"{backend_name} API request failed: {str(error)}",
            "message": "The API request encountered an error. Check your API key and internet connection.",
            "alternative": "Try using a different backend"
        }

    def _normalize_result_item(self, result, engine=None):
        """
        Normalize a single result item from any backend.

        Args:
            result: Result dict or object from backend
            engine: Optional engine name for engine-specific handling

        Returns:
            dict: Normalized result with "title", "url", "snippet"
        """
        # Handle dict results
        if isinstance(result, dict):
            title = result.get("title", "") or result.get("name", "") or result.get("product_title", "")
            url = result.get("link", "") or result.get("url", "") or result.get("href", "")
            snippet = result.get("snippet", "") or result.get("description", "") or result.get("content", "")

            # Engine-specific handling
            if engine == "youtube" and "link" in result:
                url = result.get("link", "")
                snippet = result.get("description", "") or f"Video by {result.get('channel', {}).get('name', 'Unknown')}"
            elif engine == "google_shopping" and "price" in result:
                price = result.get("price", "")
                if price:
                    snippet = f"{snippet} - {price}".strip()

            # Truncate snippet if it's from content field (Tavily/LinkUp style)
            if "content" in result and len(snippet) > 200:
                snippet = snippet[:200]

            return {"title": title, "url": url, "snippet": snippet}

        # Handle object results (LinkUp style)
        elif hasattr(result, "name"):
            return {
                "title": getattr(result, "name", ""),
                "url": getattr(result, "url", ""),
                "snippet": (getattr(result, "content", "") or getattr(result, "snippet", ""))[:200] if getattr(result, "content", None) or getattr(result, "snippet", None) else ""
            }

        # Unknown format
        raise ValueError(f"Result item is neither dict nor object: {type(result).__name__}")

    def _create_normalized_response(self, raw_response):
        """Create a normalized response structure."""
        return {
            "results": [],
            "raw_response": raw_response
        }

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
        country_code, language_code = self._get_locale_defaults(country_code, language_code, country_case="upper")

        try:
            api_key = self._check_api_key("BRAVE_API_KEY")
        except ApiKeyError as e:
            return e.error_dict

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
            return self._handle_api_request_error("Brave Search", e)

        # Normalize Brave response format
        # Brave returns: {"web": {"results": [...]}, "news": {...}, ...}
        normalized = self._create_normalized_response(data)

        # Extract web results
        web_results = data.get("web", {}).get("results", [])
        for result in web_results:
            normalized["results"].append(self._normalize_result_item(result))

        return normalized

    def _search_serper(self, query, num=10, type="search", country_code=None, language_code=None, autocorrect=True, **kwargs):
        """
        Search using Serper API (Google search) backend.

        Supports multiple search types: search, images, videos, news, shopping, places, maps, patents.

        Args:
            query (str): The search query
            num (int): Number of results to return (default: 10)
            type (str): Search type (default: "search")
                       Supported: "search", "images", "videos", "news", "shopping", "places", "maps", "patents"
                       NOTE: "scholar" is NOT supported by Serper - use serpapi with engine="google_scholar" instead
            country_code (str): 2-letter country code for localized results (e.g., "us", "gb", default: system locale)
            language_code (str): 2-letter language code for interface (e.g., "en", "es", default: system locale)
            autocorrect (bool): Whether to autocorrect the query (default: True)
            **kwargs: Additional Serper-specific parameters (location, page, tbs, etc.)

        Returns:
            Normalized dict with "results" key, or error dict
        """
        # Validate search type
        supported_types = ["search", "images", "videos", "news", "shopping", "places", "maps", "patents"]
        if type not in supported_types:
            return {
                "error": f"Unsupported search type: {type}",
                "message": f"Serper backend supports: {', '.join(supported_types)}",
                "alternative": f"For Google Scholar, use backend='serpapi' with engine='google_scholar'"
            }

        country_code, language_code = self._get_locale_defaults(country_code, language_code)

        try:
            api_key = self._check_api_key("SERPER_API_KEY")
        except ApiKeyError as e:
            return e.error_dict

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
            return self._handle_api_request_error("Serper", e)

        # Normalize Serper response format
        # Response format varies by search type
        normalized = self._create_normalized_response(data)

        # Extract results based on search type - each type has its own response structure
        serper_results_keys = {
            "search": "organic",
            "news": "news",
            "images": "images",
            "videos": "videos",
            "shopping": "shopping",
            "places": "places",
            "maps": "places",  # Maps returns places
            "patents": "organic",  # Patents returns organic-style results
        }
        results_key = serper_results_keys.get(type, "organic")

        results = data.get(results_key, [])
        for result in results:
            normalized["results"].append(self._normalize_result_item(result))

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
        country_code, language_code = self._get_locale_defaults(country_code, language_code)

        # Import SerpApi - try to get specific classes, fallback to GoogleSearch
        try:
            from serpapi import GoogleSearch
        except ImportError:
            return self._handle_import_error("google-search-results", "pip install google-search-results")

        try:
            api_key = self._check_api_key("SERPAPI_API_KEY")
        except ApiKeyError as e:
            return e.error_dict

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
                serpapi_query_params = {
                    "yahoo": "p",
                    "ebay": "_nkw",
                    "youtube": "search_query",  # YouTube uses "search_query" not "q"
                }
                search_params = {"api_key": api_key}
                query_param = serpapi_query_params.get(engine, "q")
                search_params[query_param] = query
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
            return self._handle_api_request_error("SerpApi", e)

        # Normalize SerpApi response format
        # Different engines return results in different keys
        normalized = self._create_normalized_response(data)

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

        # Check for API errors first
        if "error" in data:
            return {
                "error": f"SerpApi {engine} search error",
                "message": data.get("error", "Unknown API error"),
                "alternative": "Check your API key and query parameters"
            }

        # Get the appropriate results key for this engine
        results_key = engine_result_keys.get(engine, "organic_results")
        results = data.get(results_key, [])

        # If no results found in the expected key, fail loudly with debug info
        if not results:
            available_keys = [k for k in data.keys() if isinstance(data.get(k), list)]
            all_keys = list(data.keys())
            return {
                "error": f"No results found in expected key '{results_key}' for engine '{engine}'",
                "message": f"SerpApi response did not contain results in the expected key. Available list keys: {available_keys}. All keys: {all_keys[:20]}",
                "alternative": "Check the SerpApi response structure or try a different engine",
                "raw_response": data
            }

        # Extract and normalize results
        for result in results:
            normalized["results"].append(self._normalize_result_item(result, engine=engine))

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
            return self._handle_import_error("tavily-python", "pip install tavily-python")

        try:
            api_key = self._check_api_key("TAVILY_API_KEY")
        except ApiKeyError as e:
            return e.error_dict

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
            return self._handle_api_request_error("Tavily", e)

        # Normalize Tavily response format
        # Tavily returns: {"results": [{"title": str, "url": str, "content": str, ...}]}
        normalized = self._create_normalized_response(response)

        # Extract results
        results = response.get("results", [])
        for result in results:
            normalized["results"].append(self._normalize_result_item(result))

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
            return self._handle_import_error("linkup-sdk", "pip install linkup-sdk")

        try:
            api_key = self._check_api_key("LINKUP_API_KEY")
        except ApiKeyError as e:
            return e.error_dict

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
            return self._handle_api_request_error("LinkUp", e)

        # Normalize LinkUp response format
        # LinkUp returns a LinkupSearchResults object with .results attribute
        normalized = self._create_normalized_response(response)

        # Extract results - check if it's a list of objects or dicts
        results = getattr(response, "results", [])
        if not isinstance(results, list):
            raise ValueError(
                f"LinkUp response 'results' is not a list: {type(results).__name__}. "
                f"Response type: {type(response).__name__}"
            )

        for result in results:
            normalized["results"].append(self._normalize_result_item(result))

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

        # Define backend methods once
        backend_methods = {
            "brave": self._search_brave,
            "tavily": self._search_tavily,
            "linkup": self._search_linkup,
            "serpapi": self._search_serpapi,
            "serper": self._search_serper
        }

        if backend:
            backend = backend.lower()

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

            result = backend_methods[backend_name](query, **backend_kwargs)

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
            return self._handle_import_error("tavily-python", "pip install tavily-python")

        try:
            api_key = self._check_api_key("TAVILY_API_KEY")
        except ApiKeyError as e:
            return e.error_dict

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
            return self._handle_api_request_error("Tavily", e)

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
            return self._handle_import_error("linkup-sdk", "pip install linkup-sdk")

        try:
            api_key = self._check_api_key("LINKUP_API_KEY")
        except ApiKeyError as e:
            return e.error_dict

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
            return self._handle_api_request_error("LinkUp", e)

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
            # Use the same normalization as search results
            normalized["sources"].append(self._normalize_result_item(source))

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
        backend_methods = {
            "linkup": self._answer_linkup,
            "tavily": self._answer_tavily
        }

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue

            result = backend_methods[backend_name](question, **kwargs)

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
