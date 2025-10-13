"""
Web search utilities.

This module provides search functions for Brave and Serper APIs.

For other search providers, use their Python libraries directly:
- Tavily: `from tavily import TavilyClient` (install: tavily-python)
- SerpAPI: `from serpapi import GoogleSearch` (install: google-search-results)
- Linkup: `from linkup import LinkupClient` (install: linkup-sdk)

Example usage with external libraries:
    # Tavily
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search("your query")

    # SerpAPI
    from serpapi import GoogleSearch
    params = {"q": "your query", "api_key": os.getenv("SERPAPI_API_KEY")}
    search = GoogleSearch(params)
    results = search.get_dict()

    # Linkup
    from linkup import LinkupClient
    client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))
    results = client.search(query="your query", depth="standard", output_type="sourcedAnswer")
"""

import os
import json
import locale
import requests


class Search:
    def __init__(self, computer):
        self.computer = computer

        # Get locale-based defaults for language and country
        try:
            # Try getdefaultlocale first (returns format like 'en_US')
            loc = locale.getdefaultlocale()[0]
            if loc and '_' in loc:
                self._default_lang, self._default_country = loc.split('_')
                self._default_lang = self._default_lang.lower()
                self._default_country = self._default_country.upper()
            else:
                # Fallback to mapping common locale strings
                loc_str = str(locale.getlocale()[0] or "")
                if "English" in loc_str:
                    self._default_lang = "en"
                    if "United States" in loc_str:
                        self._default_country = "US"
                    elif "United Kingdom" in loc_str or "UK" in loc_str:
                        self._default_country = "GB"
                    else:
                        self._default_country = "US"
                else:
                    self._default_lang = "en"
                    self._default_country = "US"
        except:
            self._default_lang = "en"
            self._default_country = "US"

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
            dict: Search results from Brave API

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
            dict: Search results from Serper API

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

