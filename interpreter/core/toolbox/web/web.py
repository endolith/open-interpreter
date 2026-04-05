"""
Web search utilities.

This module provides web search related tools, with unified frontend methods
for search, fetch, answer, crawl, and structured output operations with
multiple backends.

Supported backends:
- Search: linkup, serper, serpapi, brave, tavily
- Answer: linkup, tavily
- Fetch: linkup, serper, tavily
- Crawl: tavily (not implemented yet)
- Structured output: linkup
"""

# NOTE: The first line of docstrings and their Return sections are shown to Open Interpreter in its system message, so make them very concise to avoid wasting tokens, and don't mention atypical things like error condition outputs that will confuse the AI.  Tell the AI the typical use case, and it will deal with errors when it gets to them.

import os
import json
import requests
from typing import Optional, Dict, Any

from babel import Locale
from babel.core import UnknownLocaleError


def _default_locale_from_environment() -> Locale:
    """System locale via Babel (LANG/LC_*); search APIs need ISO language + territory codes."""
    try:
        return Locale.default("LC_CTYPE")
    except (UnknownLocaleError, TypeError, ValueError, OSError):
        return Locale.parse("en_US")


def _normalize_locale_language_for_hl(lang: str) -> str:
    """ISO 639-1 for hl= / search_lang."""
    if not lang:
        return "en"
    s = lang.strip()
    if len(s) == 2 and s.isalpha():
        return s.lower()
    try:
        loc = Locale.parse(s.replace("_", "-"), sep="-")
        return (loc.language or "en").lower()
    except (UnknownLocaleError, ValueError):
        return "en"


def _normalize_locale_country_for_gl(country: str) -> str:
    """ISO 3166-1 alpha-2 (uppercase) for gl= / country."""
    if not country:
        return "US"
    s = " ".join(country.replace("\u00a0", " ").split()).strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    try:
        loc = Locale.parse(s.replace("_", "-"), sep="-")
        if loc.territory:
            return loc.territory.upper()
    except (UnknownLocaleError, ValueError):
        pass
    return "US"


class ApiKeyError(Exception):
    """Exception raised when an API key is missing. Contains error dict."""
    def __init__(self, error_dict):
        self.error_dict = error_dict
        super().__init__(error_dict.get("error", "API key missing"))

    def _render_traceback_(self):
        # Suppress the full Jupyter/IPython traceback — just show the message.
        msg = self.error_dict.get("message", str(self))
        return [f"ApiKeyError: {msg}"]


class WebToolboxError(Exception):
    """Raised when a web toolbox operation fails (missing package, no backends, API error, etc.)."""

    def _render_traceback_(self):
        # Suppress the full Jupyter/IPython traceback — just show the message.
        return [f"WebToolboxError: {self}"]


class SearchResult(dict):
    """Dict subclass for web search results. Has a compact repr to avoid flooding the context window."""

    def __init__(self, data, web=None):
        super().__init__(data)
        self._web = web

    def fetch(self, index):
        """Fetch the full page for search result at the given index. Returns a FetchResult."""
        results = self.get("results", [])
        url = results[index]["url"]
        return self._web.fetch(url)

    def __repr__(self):
        backend = self.get("backend", "?")
        results = self.get("results", [])
        n = len(results)
        lines = [f"SearchResult({n} results) [backend={backend}]"]
        lines.append("  Keys: results[list of {title,url,snippet}], raw_response[dict], backend[str]")
        lines.append("  → result['results'][i] | result.fetch(i)")
        for i, r in enumerate(results[:5]):
            title = r.get("title", "")[:70]
            url = r.get("url", "")
            domain = url.split("/")[2] if url.count("/") >= 2 else url
            snippet = r.get("snippet", "")[:120]
            lines.append(f"  {i}. \"{title}\" \u2014 {domain}")
            if snippet:
                lines.append(f"     {snippet}")
        if n > 5:
            lines.append(f"  ... {n - 5} more")
        return "\n".join(lines)


class FetchResult(dict):
    """Dict subclass for web fetch results. Has a compact repr to avoid flooding the context window."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached = False

    def _get_content(self):
        """Return content string. For multi-page results, concatenates all pages."""
        if "content" in self:
            return self["content"]
        return "\n\n".join(r.get("content", "") for r in self.get("results", []))

    def find(self, term, context=100, max_results=None):
        """
        Find all occurrences of term in content (case-insensitive).
        Returns a list of snippet strings, each with up to `context` chars of surrounding text.
        Pass max_results to cap the number of matches returned.
        """
        import re
        content = self._get_content()
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        snippets = []
        for m in pattern.finditer(content):
            start = max(0, m.start() - context)
            end = min(len(content), m.end() + context)
            snippets.append(content[start:end].replace("\n", " ").strip())
            if max_results is not None and len(snippets) >= max_results:
                break
        return snippets

    def links(self):
        """
        Extract hyperlinks from content.
        Returns a list of (anchor_text, url) tuples parsed from markdown [text](url) syntax.
        """
        import re
        content = self._get_content()
        return re.findall(r'\[([^\]]*)\]\((https?://[^)]+)\)', content)

    def __repr__(self):
        backend = self.get("backend", "?")
        cached_tag = " [cached]" if getattr(self, "_cached", False) else ""
        if "results" in self:
            # Multi-URL result from explicit urls=[...] kwarg (tavily only)
            results = self.get("results", [])
            n = len(results)
            lines = [f"FetchResult({n} pages) [backend={backend}]{cached_tag}"]
            lines.append("  Keys: results[list of {url,title,content}], raw_response[dict], backend[str]")
            lines.append("  → page=result.fetch(i) → page.find()|page.links()|page['results'][i]['content']")
            for r in results[:3]:
                title = r.get("title", "")[:50]
                url = r.get("url", "")
                domain = url.split("/")[2] if url.count("/") >= 2 else url
                content_len = len(r.get("content", ""))
                lines.append(f"  \u2022 \"{title}\" \u2014 {domain} ({content_len:,} chars)")
            if n > 3:
                lines.append(f"  ... {n - 3} more")
        else:
            # Single-page result (all backends for single-URL fetch)
            title = self.get("title", "")
            content = self.get("content", "")
            content_len = len(content)
            preview = content[:150].replace("\n", " ") if content else ""
            extra_keys = ", ".join(
                f"{k}[dict]" for k in self.keys()
                if k not in ("title", "url", "content", "backend")
            )
            lines = [f"FetchResult [backend={backend}]{cached_tag}"]
            lines.append(
                f"  Keys: url[str], title[str], content[str={content_len:,} chars]"
                + (f", {extra_keys}" if extra_keys else "")
                + ", backend[str]"
            )
            lines.append("  → page=result.fetch(i) → page.find()|page.links()")
            if title:
                lines.append(f"  \"{title}\"")
            else:
                lines.append("  [no title]")
            if preview:
                lines.append(f"  {preview}...")
        return "\n".join(lines)


class AnswerResult(dict):
    """Dict subclass for web answer results. Has a compact repr to avoid flooding the context window."""

    def __init__(self, data, web=None):
        super().__init__(data)
        self._web = web

    def fetch(self, index):
        """Fetch the full page for source at the given index. Returns a FetchResult."""
        sources = self.get("sources", [])
        url = sources[index]["url"]
        return self._web.fetch(url)

    def __repr__(self):
        backend = self.get("backend", "?")
        answer = self.get("answer", "")
        sources = self.get("sources", [])
        n_sources = len(sources)
        lines = [f"AnswerResult({n_sources} sources) [backend={backend}]"]
        lines.append("  Keys: answer[str], sources[list of {title,url,snippet}], backend[str]")
        lines.append("  → result['answer']|result['sources'][i]|result.fetch(i)")
        if answer:
            for line in answer.split("\n"):
                lines.append(f"  {line}")
        return "\n".join(lines)


class StructuredOutputResult(dict):
    """Dict subclass for web search results with structured output (JSON)."""

    def __init__(self, data, web=None):
        super().__init__(data)
        self._web = web

    def fetch(self, index):
        """Fetch the full page for source at the given index. Returns a FetchResult."""
        sources = self.get("sources", [])
        if not sources:
            raise WebToolboxError("No sources available in this result to fetch.")
        url = sources[index]["url"]
        return self._web.fetch(url)

    def __repr__(self):
        backend = self.get("backend", "?")
        data = self.get("structured_output", {})
        # Show some of the fields to be helpful but not flood repr
        keys = list(data.keys()) if isinstance(data, dict) else []
        lines = [f"StructuredOutputResult({len(keys)} top-level keys) [backend={backend}]"]
        lines.append(f"  Keys: {', '.join(keys[:10])}{'...' if len(keys) > 10 else ''}")
        # Pretty print a bit of JSON as preview — use 2-space indent, max 6 lines
        try:
            preview = json.dumps(data, indent=2)
            for line in preview.splitlines()[:6]:
                lines.append(f"  {line}")
            if len(preview.splitlines()) > 6:
                lines.append("  ...")
        except (TypeError, ValueError):
            lines.append(f"  {str(data)[:200]}...")
        return "\n".join(lines)


def _normalize_tavily_single_page(result):
    """
    Tavily's extract API always returns a list, even for a single URL.
    Unwrap it to the flat {url, title, content, raw_response} structure
    that serper and linkup return, so FetchResult is consistent across backends.
    Only used for single-URL fetches; multi-URL calls keep the list structure.
    """
    pages = result.get("results", [])
    if not pages:
        raise WebToolboxError(
            "Tavily returned no results for this URL. "
            "The page may be inaccessible or blocked. Try a different backend."
        )
    flat = pages[0].copy()
    flat["raw_response"] = result.get("raw_response", {})
    return flat


class Web:
    def __init__(self, toolbox):
        self.toolbox = toolbox
        _loc = _default_locale_from_environment()
        self._default_lang = (_loc.language or "en").lower()
        self._default_country = (_loc.territory or "US").upper()
        # Session-scoped cache: keyed by URL. Web page content doesn't change
        # mid-session, so re-fetching the same URL is always wasteful.
        self._fetch_cache: Dict[str, "FetchResult"] = {}

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
            country_code = self._default_country
        else:
            country_code = _normalize_locale_country_for_gl(country_code)
        if language_code is None:
            language_code = self._default_lang
        else:
            language_code = _normalize_locale_language_for_hl(language_code)
        if country_case == "lower":
            country_code = country_code.lower()
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
        """Raise WebToolboxError when a required package is not installed."""
        raise WebToolboxError(f"Install {package_name}: {install_cmd}")

    def _handle_api_request_error(self, backend_name, error):
        """Raise WebToolboxError for API request failures."""
        raise WebToolboxError(
            f"{backend_name} API request failed: {error}. Check your API key and internet connection."
        )

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
            Normalized dict with "results" key
        """
        country_code, language_code = self._get_locale_defaults(country_code, language_code, country_case="upper")

        try:
            api_key = self._check_api_key("BRAVE_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        url = "https://api.search.brave.com/res/v1/web/search"

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }

        # kwargs last would let stray keys (e.g. country=...) override normalized locale — merge first.
        params = {
            **kwargs,
            "q": query,
            "count": min(count, 20),  # Brave has a max of 20
            "country": country_code,
            "search_lang": language_code,
            "safesearch": safesearch,
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self._handle_api_request_error("Brave Search", e)

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
            Normalized dict with "results" key
        """
        # Validate search type
        supported_types = ["search", "images", "videos", "news", "shopping", "places", "maps", "patents"]
        if type not in supported_types:
            raise WebToolboxError(
                f"Serper backend supports: {', '.join(supported_types)}. "
                f"For Google Scholar, use backend='serpapi' with engine='google_scholar'"
            )

        country_code, language_code = self._get_locale_defaults(country_code, language_code)

        try:
            api_key = self._check_api_key("SERPER_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        # Map type to correct endpoint
        url = f"https://google.serper.dev/{type}"

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }

        payload = {
            **kwargs,
            "q": query,
            "num": num,
            "gl": country_code,
            "hl": language_code,
            "autocorrect": autocorrect,
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self._handle_api_request_error("Serper", e)

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
            Normalized dict with "results" key
        """
        country_code, language_code = self._get_locale_defaults(country_code, language_code)

        # Import SerpApi - try to get specific classes, fallback to GoogleSearch
        try:
            from serpapi import GoogleSearch
        except ImportError:
            self._handle_import_error("google-search-results", "pip install google-search-results")

        try:
            api_key = self._check_api_key("SERPAPI_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

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
                    **kwargs,
                    "q": query,
                    "num": num,
                    "engine": engine,
                    "api_key": api_key,
                    "gl": country_code,
                    "hl": language_code,
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
                    raise WebToolboxError(
                        f"The {class_name} class is not available in your serpapi package version. "
                        "Update: pip install --upgrade google-search-results"
                    )

                # Build params - different engines use different query parameter names
                serpapi_query_params = {
                    "yahoo": "p",
                    "ebay": "_nkw",
                    "youtube": "search_query",  # YouTube uses "search_query" not "q"
                }
                search_params = {**kwargs, "api_key": api_key}
                query_param = serpapi_query_params.get(engine, "q")
                search_params[query_param] = query
                if num:
                    search_params["num"] = num

                # Add localization for engines that support it (after kwargs so locale wins)
                if engine in ["bing", "yahoo", "duckduckgo", "youtube"]:
                    search_params["gl"] = country_code
                    search_params["hl"] = language_code

                search = SearchClass(search_params)
                data = search.get_dict()
            else:
                raise WebToolboxError(
                    f"Engine '{engine}' is not supported. Supported engines: "
                    f"{', '.join(google_engines + list(specific_class_engines.keys()))}"
                )
        except Exception as e:
            self._handle_api_request_error("SerpApi", e)

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
            raise WebToolboxError(
                f"SerpApi {engine} search error: {data.get('error', 'Unknown API error')}. "
                "Check your API key and query parameters."
            )

        # Get the appropriate results key for this engine
        results_key = engine_result_keys.get(engine, "organic_results")
        results = data.get(results_key, [])

        # If no results found in the expected key, fail loudly with debug info
        if not results:
            available_keys = [k for k in data.keys() if isinstance(data.get(k), list)]
            all_keys = list(data.keys())
            raise WebToolboxError(
                f"SerpApi response did not contain results in expected key '{results_key}'. "
                f"Available list keys: {available_keys}. All keys: {all_keys[:20]}"
            )

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
            Normalized dict with "results" key
        """
        # Tavily doesn't use country/language codes directly, but we accept them for consistency
        try:
            from tavily import TavilyClient
        except ImportError:
            self._handle_import_error("tavily-python", "pip install tavily-python")

        try:
            api_key = self._check_api_key("TAVILY_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

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
            self._handle_api_request_error("Tavily", e)

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
            Normalized dict with "results" key
        """
        # LinkUp doesn't use country/language codes directly, but we accept them for consistency
        try:
            from linkup import LinkupClient
        except ImportError:
            self._handle_import_error("linkup-sdk", "pip install linkup-sdk")

        try:
            api_key = self._check_api_key("LINKUP_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

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
            self._handle_api_request_error("LinkUp", e)

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

    def _build_no_backends_error(self, backends_to_try, failed_results, backend_to_package, backend_to_key, kind):
        """
        Build error message when no backends succeeded. failed_results is a list of
        (backend_name, exception). Returns a string message for WebToolboxError.
        """
        failed_by_backend = {b: exc for b, exc in failed_results}
        reasons = []
        for b in backends_to_try:
            if not self._check_backend_available(b):
                key_name = backend_to_key.get(b, b.upper() + "_API_KEY")
                reasons.append((b, f"API key not set (set {key_name})"))
            elif b in failed_by_backend:
                exc = failed_by_backend[b]
                err_str = str(exc)
                if isinstance(exc, ApiKeyError):
                    msg = exc.error_dict.get("message", err_str)
                    reasons.append((b, "API key not set (" + msg + ")"))
                elif "not installed" in err_str or "Install " in err_str:
                    pkg = backend_to_package.get(b, b)
                    reasons.append((b, f"package not installed (pip install {pkg})"))
                else:
                    # Surface the actual backend error message so callers can see
                    # why each backend failed instead of a generic placeholder.
                    # Flatten multi-line errors to keep the aggregate message readable.
                    clean_err = " ".join(err_str.splitlines()).strip()
                    if isinstance(exc, WebToolboxError):
                        # WebToolboxError messages are already user-oriented; use as-is.
                        reasons.append((b, clean_err or "request failed"))
                    else:
                        # For other exception types, include them as a failure reason.
                        reasons.append(
                            (b, f"request failed: {clean_err}" if clean_err else "request failed")
                        )
            else:
                reasons.append((b, "unavailable"))
        kind_label = f"{kind} " if kind else ""
        return f"No {kind_label}backends are working. " + ". ".join(f"{b}: {msg}" for b, msg in reasons)

    def search(self, query: str, backend: Optional[str] = None, country_code: Optional[str] = None, language_code: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Search the web for links and snippets.

        This method automatically selects the best available backend or uses
        the specified one. Backends are tried in order: serper, serpapi, tavily, brave, linkup.

        Args:
            query (str): The search query
            backend (str, optional): Force a specific backend ("brave", "tavily", "linkup", "serpapi", or "serper").
                                     If None, auto-selects based on availability.
            country_code (str, optional): 2-letter country code for localized results (e.g., "US", "GB", "FR").
                                          Defaults to system locale. Supported by: brave, serpapi, serper.
            language_code (str, optional): 2-letter language code for search interface (e.g., "en", "es", "fr").
                                           Defaults to system locale. Supported by: brave, serpapi, serper.
            **kwargs: Additional backend-specific parameters:

                BRAVE:
                    - count (int, 1-20): Number of results (default: 10, max: 20)
                    - safesearch (str): "off", "moderate", or "strict" (default: "moderate")
                    - freshness (str): "pd" (past day), "pw" (past week), "pm" (past month), "py" (past year)
                    - text_decorations (bool): Include text decorations in snippets (default: True)
                    - spellcheck (bool): Enable spellcheck (default: True)
                    NOTE: Use country_code and language_code parameters (not country/search_lang)

                TAVILY:
                    - max_results (int): Number of results to return (default: 10)
                    - search_depth (str): "basic" or "advanced" (default: "basic")
                    - include_domains (list): List of domains to include (e.g., ["example.com"])
                    - exclude_domains (list): List of domains to exclude
                    - include_raw_content (bool): Include full HTML content (default: False)
                    - include_images (bool): Include images in results (default: False)
                    - topic (str): "general" or "news" (default: "general")
                    - days (int): Number of days back to search (for topic="news")

                LINKUP:
                    - depth (str): "standard" or "deep" (default: "standard")
                    - output_type (str): "searchResults" (default), "sourcedAnswer", or "structured"
                    - structured_output_schema (dict): Schema for structured output mode

                SERPAPI - supports 80+ search engines:
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

                SERPER - supports multiple search types:
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
            SearchResult: {"results": list, "raw_response": dict, "backend": str}

        Examples:
            # Basic search (auto-selects backend)
            results = toolbox.web.search("machine learning tutorials")
            for result in results["results"]:
                print(f"{result['title']}: {result['url']}")

            # Search Google Scholar for academic papers (using serpapi)
            results = toolbox.web.search(
                "quantum computing",
                backend="serpapi",
                engine="google_scholar"
            )

            # Search YouTube videos (using serpapi)
            results = toolbox.web.search(
                "python tutorial",
                backend="serpapi",
                engine="youtube"
            )

            # Search Google News (using serper)
            results = toolbox.web.search(
                "AI breakthroughs",
                backend="serper",
                type="news"
            )

            # Deep web search with specific domains (using tavily)
            results = toolbox.web.search(
                "climate change research",
                backend="tavily",
                search_depth="advanced",
                include_domains=["nature.com", "science.org"]
            )

            # Shopping search (using serpapi)
            results = toolbox.web.search(
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
                raise WebToolboxError(
                    f"Supported backends for search: {', '.join(backend_methods.keys())}. "
                    "Try without specifying a backend to auto-select."
                )

            result = backend_methods[backend](query, **backend_kwargs)
            result["backend"] = backend
            print("→ result['results'][i]|result.fetch(i)")
            return SearchResult(result, web=self)

        # Auto-select backend
        # Priority order based on AI agent needs: serper (rich snippets, knowledge panels, structured data, best for AI) > serpapi (comprehensive Google results) > tavily (AI-optimized) > brave (alternative sources, fewer snippets) > linkup
        backends_to_try = ["serper", "serpapi", "tavily", "brave", "linkup"]
        failed_results = []

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue
            try:
                result = backend_methods[backend_name](query, **backend_kwargs)
                result["backend"] = backend_name
                print("→ result['results'][i]|result.fetch(i)")
                return SearchResult(result, web=self)
            except (WebToolboxError, ApiKeyError) as e:
                failed_results.append((backend_name, e))

        search_backend_to_package = {"serper": "google-search-results (serper)", "serpapi": "google-search-results", "tavily": "tavily-python", "brave": "brave-search-sdk", "linkup": "linkup-sdk"}
        search_backend_to_key = {"serper": "SERPER_API_KEY", "serpapi": "SERPAPI_API_KEY", "tavily": "TAVILY_API_KEY", "brave": "BRAVE_API_KEY", "linkup": "LINKUP_API_KEY"}
        message = self._build_no_backends_error(
            backends_to_try, failed_results, search_backend_to_package, search_backend_to_key, kind="search"
        )
        raise WebToolboxError(message)

    def _answer_tavily(self, question, answer_mode="basic", **kwargs):
        """
        Get AI-generated answer using Tavily backend.

        Args:
            question: The question to answer
            answer_mode: "basic" or "advanced" (default: "basic")
            **kwargs: Additional Tavily search parameters

        Returns:
            Normalized dict with "answer" and "sources" keys
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            self._handle_import_error("tavily-python", "pip install tavily-python")

        try:
            api_key = self._check_api_key("TAVILY_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

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
            self._handle_api_request_error("Tavily", e)

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
            Normalized dict with "answer" and "sources" keys
        """
        try:
            from linkup import LinkupClient
        except ImportError:
            self._handle_import_error("linkup-sdk", "pip install linkup-sdk")

        try:
            api_key = self._check_api_key("LINKUP_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

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
            self._handle_api_request_error("LinkUp", e)

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

    def _structured_output_linkup(self, query, structured_output_schema, depth="standard", **kwargs):
        """
        Get JSON structured output using LinkUp backend.

        Args:
            query: The search query
            structured_output_schema: Dict representing JSON schema
            depth: "standard" or "deep" (default: "standard")
            **kwargs: Additional LinkUp search parameters

        Returns:
            Normalized dict with "structured_output" and optionally "sources"
        """
        try:
            from linkup import LinkupClient
        except ImportError:
            self._handle_import_error("linkup-sdk", "pip install linkup-sdk")

        try:
            api_key = self._check_api_key("LINKUP_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        try:
            client = LinkupClient(api_key=api_key)

            # Build search parameters
            search_params = {
                "query": query,
                "depth": depth,
                "output_type": "structured",
                "structured_output_schema": structured_output_schema,
                **kwargs
            }

            response = client.search(**search_params)
        except Exception as e:
            self._handle_api_request_error("LinkUp", e)

        # LinkUp returns a LinkupStructuredOutput object with .structured_output attribute
        # and .sources attribute.
        structured_data = getattr(response, "structured_output", response)

        # If response was a dict and had specific keys (e.g. if SDK version changes)
        if isinstance(structured_data, dict) and "structured_output" in structured_data:
            structured_data = structured_data["structured_output"]

        normalized = {
            "structured_output": structured_data,
            "sources": []
        }

        # Check if sources are available
        sources = getattr(response, "sources", [])
        if sources and isinstance(sources, list):
            for source in sources:
                normalized["sources"].append(self._normalize_result_item(source))

        return normalized

    def answer(self, question: str, backend: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        AI-synthesized answer from web sources. PREFERRED for direct questions about current events

        This method automatically selects the best available backend or uses
        the specified one. Backends are tried in order: linkup, tavily.

        Args:
            question (str): MUST be a natural-language question ending in "?", e.g. "What is the best vacuum cleaner in 2026?" — NOT search terms like "best vacuum cleaner 2026". An AI agent reads the web and answers the question for you.
            backend (str, optional): Force a specific backend ("tavily" or "linkup").
                                     If None, auto-selects based on availability.
            **kwargs: Additional backend-specific parameters:
                - For tavily: answer_mode ("basic" or "advanced"), search_depth, etc.
                - For linkup: depth ("standard" or "deep"), include_inline_citations, etc.

        Returns:
            AnswerResult: {"answer": str, "sources": list, "backend": str}

        Example:
            result = toolbox.web.answer("What is the latitude of Lilongwe in decimal format?")
            print(result["answer"])
            for source in result["sources"]:
                print(f"- {source['title']}: {source['url']}")
        """
        if "?" not in question:
            print(f"⚠️  web.answer() expects a question ending in '?', not search terms.\n"
                  f"   Try rephrasing: \"{question}?\"")

        if backend:
            backend = backend.lower()
            if backend not in ("tavily", "linkup"):
                raise WebToolboxError(
                    "Supported backends for answer: 'tavily', 'linkup'. Try without specifying a backend to auto-select."
                )
            backend_methods = {"linkup": self._answer_linkup, "tavily": self._answer_tavily}
            result = backend_methods[backend](question, **kwargs)
            result["backend"] = backend
            print("→ result['answer']|result['sources']|result.fetch(i)")
            return AnswerResult(result, web=self)

        backends_to_try = ["linkup", "tavily"]
        backend_methods = {
            "linkup": self._answer_linkup,
            "tavily": self._answer_tavily
        }
        failed_results = []

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue
            try:
                result = backend_methods[backend_name](question, **kwargs)
                result["backend"] = backend_name
                print("→ result['answer']|result['sources']|result.fetch(i)")
                return AnswerResult(result, web=self)
            except (WebToolboxError, ApiKeyError) as e:
                failed_results.append((backend_name, e))

        message = self._build_no_backends_error(
            backends_to_try,
            failed_results,
            backend_to_package={"linkup": "linkup-sdk", "tavily": "tavily-python"},
            backend_to_key={"linkup": "LINKUP_API_KEY", "tavily": "TAVILY_API_KEY"},
            kind="answer"
        )
        raise WebToolboxError(message)

    def structured_output(self, query: str, schema: Any, backend: Optional[str] = "linkup", **kwargs) -> Dict[str, Any]:
        """
        Search and extract specific fields defined by schema (dict or Pydantic). PREFERRED for data extraction.

        This method is best for tasks requiring extracting specific fields (like author, year, title)
        directly from web resources into a schema-defined format.

        Args:
            query (str): The search query or data extraction prompt.
            schema (dict or Pydantic model): The JSON schema defining the desired output structure.
            backend (str, optional): Force a specific backend (default: "linkup").
            **kwargs: Additional backend-specific parameters:
                - For linkup: depth ("standard" or "deep"), etc.

        Returns:
            StructuredOutputResult: {"structured_output": dict, "sources": list, "backend": str}

        Example:
            # Using journal article schema
            schema = {
                "type": "object",
                "properties": {
                    "author_last_name": {"type": "string", "description": "Last name of the first author"},
                    "year": {"type": "integer", "description": "Year of publication"},
                    "title": {"type": "string", "description": "Full title of the article"}
                },
                "required": ["author_last_name", "year", "title"]
            }
            result = toolbox.web.structured_output("Attention is All You Need journal article", schema=schema)
            print(result["structured_output"]["author_last_name"])
        """
        import json

        # LinkUp SDK expects a Pydantic model CLASS or a JSON STRING or None.
        # It does NOT accept a dictionary directly.
        is_pydantic = False
        try:
            # Check if it's a Pydantic class (v1 or v2)
            if isinstance(schema, type):
                # Try to import any version of Pydantic to check inheritance
                try:
                    from pydantic import BaseModel as BM2
                    if issubclass(schema, BM2):
                        is_pydantic = True
                except ImportError:
                    pass

                if not is_pydantic:
                    try:
                        from pydantic.v1 import BaseModel as BM1
                        if issubclass(schema, BM1):
                            is_pydantic = True
                    except ImportError:
                        pass
            elif hasattr(schema, "__pydantic_model__"): # some wrappers
                 is_pydantic = True
        except Exception:
            # If any check fails, treat as non-pydantic
            pass

        if not is_pydantic and isinstance(schema, dict):
            # Convert dictionary to JSON string as expected by the LinkUp SDK
            schema = json.dumps(schema)
        # If it is a Pydantic class or already a string, pass it through to the backend.

        if backend:
            backend = backend.lower()
            if backend != "linkup":
                 raise WebToolboxError(
                    "Only LinkUp currently supports structured output via backend='linkup'."
                )
            backend_methods = {"linkup": self._structured_output_linkup}
            result = backend_methods[backend](query, schema, **kwargs)
            result["backend"] = backend
            print("→ result['structured_output']|result['sources']|result.fetch(i)")
            return StructuredOutputResult(result, web=self)

        # Default/Auto-select (currently only linkup)
        backends_to_try = ["linkup"]
        backend_methods = {"linkup": self._structured_output_linkup}
        failed_results = []

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue
            try:
                result = backend_methods[backend_name](query, schema, **kwargs)
                result["backend"] = backend_name
                print("→ result['structured_output']|result['sources']|result.fetch(i)")
                return StructuredOutputResult(result, web=self)
            except (WebToolboxError, ApiKeyError) as e:
                failed_results.append((backend_name, e))

        message = self._build_no_backends_error(
            backends_to_try,
            failed_results,
            backend_to_package={"linkup": "linkup-sdk"},
            backend_to_key={"linkup": "LINKUP_API_KEY"},
            kind="structured output"
        )
        raise WebToolboxError(message)

    def _fetch_serper(self, url, **kwargs):
        """
        Fetch web page content using Serper API (scrape.serper.dev).

        Args:
            url (str): The URL to fetch
            **kwargs: Additional Serper-specific parameters

        Returns:
            Normalized dict with "content" (markdown), "title", "url" keys
        """
        try:
            api_key = self._check_api_key("SERPER_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        # Serper scrape endpoint
        scrape_url = "https://scrape.serper.dev"

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }

        # Ensure markdown output (default for Serper)
        payload = {
            "url": url,
            "markdown": True,
            **kwargs
        }

        try:
            response = requests.post(scrape_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self._handle_api_request_error("Serper", e)

        # Normalize Serper response format - always return markdown
        # Serper returns: {"markdown": str, "text": str, "metadata": {"title": str}, ...}
        # Title can be in metadata.title or top-level title
        title = data.get("title", "") or data.get("metadata", {}).get("title", "")

        normalized = {
            "url": url,
            "title": title,
            "content": data.get("markdown", "") or data.get("text", ""),  # Prefer markdown, fallback to text
            "raw_response": data
        }

        return normalized

    def _fetch_linkup(self, url, render_js=False, **kwargs):
        """
        Fetch web page content using LinkUp backend.

        Args:
            url (str): The URL to fetch
            render_js (bool): Whether to render JavaScript (default: False)
            **kwargs: Additional LinkUp fetch parameters

        Returns:
            Normalized dict with "content" (markdown), "title", "url" keys
        """
        try:
            from linkup import LinkupClient
        except ImportError:
            self._handle_import_error("linkup-sdk", "pip install linkup-sdk")

        try:
            api_key = self._check_api_key("LINKUP_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        try:
            client = LinkupClient(api_key=api_key)

            # Build fetch parameters
            # LinkUp fetch() returns markdown by default, no output_format parameter needed
            fetch_params = {
                "url": url,
                "render_js": render_js,
                **kwargs
            }

            response = client.fetch(**fetch_params)
        except Exception as e:
            self._handle_api_request_error("LinkUp", e)

        # LinkUp returns a LinkupFetchResult object or dict
        # LinkUp fetch endpoint does NOT provide title as a separate field
        # Always extract markdown content (prefer markdown, fallback to html if needed)
        if hasattr(response, "markdown"):
            content = getattr(response, "markdown", "") or getattr(response, "html", "")
        elif isinstance(response, dict):
            content = response.get("markdown", "") or response.get("html", "")
        else:
            content = str(response)

        # Title is not provided by LinkUp backend
        title = ""

        normalized = {
            "url": url,
            "title": title,
            "content": content,  # Always markdown (normalized)
            "raw_response": response
        }

        return normalized

    def _fetch_tavily(self, urls, extract_depth=None, **kwargs):
        """
        Fetch web page content using Tavily extract endpoint.

        Args:
            urls (str or list): Single URL string or list of URLs (max 20)
            extract_depth (str): "basic" or "advanced" (default: "basic")
            **kwargs: Additional Tavily extract parameters

        Returns:
            Normalized dict with "results" list (each with "content" (markdown), "title", "url")
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            self._handle_import_error("tavily-python", "pip install tavily-python")

        try:
            api_key = self._check_api_key("TAVILY_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        # Normalize urls to list
        if isinstance(urls, str):
            urls = [urls]
        elif not isinstance(urls, list):
            raise WebToolboxError(
                "urls must be a string (single URL) or list of URLs (max 20). Pass a single URL string or a list of URLs."
            )

        if len(urls) > 20:
            raise WebToolboxError(
                "Tavily extract endpoint supports maximum 20 URLs per request. Split URLs into multiple requests of 20 or fewer."
            )

        try:
            client = TavilyClient(api_key=api_key)

            # Build extract parameters
            # According to Tavily docs, format defaults to "markdown" which is what we want
            extract_params = {"urls": urls, "format": "markdown"}
            if extract_depth is not None:
                extract_params["extract_depth"] = extract_depth
            extract_params.update(kwargs)

            response = client.extract(**extract_params)
        except Exception as e:
            self._handle_api_request_error("Tavily", e)

        # Tavily returns: {"results": [{"url": str, "title": str, "content": str, ...}, ...]}
        if not isinstance(response, dict):
            raise ValueError(
                f"Tavily returned unexpected response type: {type(response).__name__}. "
                f"Expected dict, got {type(response).__name__}. "
                f"Response: {str(response)[:500]}"
            )

        normalized = {
            "results": [],
            "raw_response": response
        }

        results = response.get("results", [])
        if not isinstance(results, list):
            raise ValueError(
                f"Tavily response missing or invalid 'results' field. "
                f"Expected list, got {type(results).__name__}. "
                f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}"
            )

        # Check for failed results
        failed_results = response.get("failed_results", [])
        if failed_results and not results:
            failed_urls = [fr.get("url", "unknown") for fr in failed_results if isinstance(fr, dict)]
            errors = [fr.get("error", "unknown error") for fr in failed_results if isinstance(fr, dict)]
            raise WebToolboxError(
                f"Tavily extract failed for all URLs. Failed: {', '.join(failed_urls[:3])}. "
                f"Errors: {', '.join(errors[:3])}. Try a different backend or check if the URLs are accessible."
            )

        for result in results:
            if not isinstance(result, dict):
                raise ValueError(
                    f"Tavily result item is not a dict: {type(result).__name__}. "
                    f"Result: {str(result)[:200]}"
                )
            normalized["results"].append({
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "content": result.get("raw_content", "") or result.get("content", "")  # Tavily returns "raw_content"
            })

        # If some URLs failed but we have some results, include failed_results in raw_response
        # (already included, but we could add a warning if needed)

        return normalized

    def fetch(self, url: str, backend: Optional[str] = None, render_js: bool = False, extract_depth: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Fetch web page content from a URL as markdown.

        This method automatically selects the best available backend or uses
        the specified one. Backends are tried in order: serper, linkup, tavily.

        Args:
            url (str): The URL to fetch
            backend (str, optional): Force a specific backend ("serper", "linkup", or "tavily").
                                     If None, auto-selects based on availability.
            render_js (bool): Whether to render JavaScript (default: False). Supported by: linkup
            extract_depth (str, optional): Extraction depth - "basic" or "advanced". Supported by: tavily (defaults to API default if not specified)
            **kwargs: Additional backend-specific parameters:

                SERPER:
                    - Other Serper scrape parameters (markdown is always enabled)

                LINKUP:
                    - Other LinkUp fetch parameters (output is always markdown)

                TAVILY:
                    - urls (list): List of URLs to fetch (max 20). If provided, overrides url parameter.
                    - include_images (bool): Include images in extraction (default: False)
                    - Other Tavily extract parameters

        Returns:
            FetchResult: {"url": str, "title": str, "content": str, "raw_response": dict, "backend": str}

        Examples:
            # Basic fetch (auto-selects backend)
            result = toolbox.web.fetch("https://example.com")
            print(result["title"])
            print(result["content"][:500])

            # Fetch with JavaScript rendering
            result = toolbox.web.fetch(
                "https://example.com",
                render_js=True
            )

            # Fetch with advanced extraction depth
            result = toolbox.web.fetch(
                "https://example.com",
                extract_depth="advanced"
            )

            # Fetch multiple pages at once (using tavily)
            result = toolbox.web.fetch(
                "https://example.com",
                backend="tavily",
                urls=["https://example.com", "https://example.org"]
            )
        """
        # Define backend methods
        backend_methods = {
            "serper": self._fetch_serper,
            "linkup": self._fetch_linkup,
            "tavily": self._fetch_tavily
        }

        # Multi-URL calls (tavily urls=[...]) bypass the cache — too varied to key simply.
        is_multi_url = "urls" in kwargs

        if not is_multi_url and url in self._fetch_cache:
            cached = self._fetch_cache[url]
            cached._cached = True
            print("→ page=result.fetch(i) → page.find()|page.links()")
            return cached

        if backend:
            backend = backend.lower()

            if backend not in backend_methods:
                raise WebToolboxError(
                    f"Supported backends for fetch: {', '.join(backend_methods.keys())}. "
                    "Try without specifying a backend to auto-select."
                )

            if is_multi_url:
                result = backend_methods[backend](kwargs["urls"], extract_depth=extract_depth, **{k: v for k, v in kwargs.items() if k != "urls"})
            elif backend == "tavily":
                result = backend_methods[backend]([url], extract_depth=extract_depth, **kwargs)
                result = _normalize_tavily_single_page(result)
            elif backend == "linkup":
                result = backend_methods[backend](url, render_js=render_js, **kwargs)
            else:
                result = backend_methods[backend](url, **kwargs)

            result["backend"] = backend
            fetch_result = FetchResult(result)
            if not is_multi_url:
                self._fetch_cache[url] = fetch_result
            print("→ page=result.fetch(i) → page.find()|page.links()")
            return fetch_result

        backends_to_try = ["serper", "linkup", "tavily"]
        failed_results = []

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue
            try:
                if is_multi_url:
                    result = backend_methods[backend_name](kwargs["urls"], extract_depth=extract_depth, **{k: v for k, v in kwargs.items() if k != "urls"})
                elif backend_name == "tavily":
                    result = backend_methods[backend_name]([url], extract_depth=extract_depth, **kwargs)
                    result = _normalize_tavily_single_page(result)
                elif backend_name == "linkup":
                    result = backend_methods[backend_name](url, render_js=render_js, **kwargs)
                else:
                    result = backend_methods[backend_name](url, **kwargs)
                result["backend"] = backend_name
                fetch_result = FetchResult(result)
                if not is_multi_url:
                    self._fetch_cache[url] = fetch_result
                print("→ page=result.fetch(i) → page.find()|page.links()")
                return fetch_result
            except (WebToolboxError, ApiKeyError) as e:
                failed_results.append((backend_name, e))

        fetch_backend_to_package = {"serper": "requests (built-in)", "linkup": "linkup-sdk", "tavily": "tavily-python"}
        fetch_backend_to_key = {"serper": "SERPER_API_KEY", "linkup": "LINKUP_API_KEY", "tavily": "TAVILY_API_KEY"}
        message = self._build_no_backends_error(
            backends_to_try, failed_results, fetch_backend_to_package, fetch_backend_to_key, kind="fetch"
        )
        raise WebToolboxError(message)
