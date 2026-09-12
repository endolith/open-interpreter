import os
import unittest
import json
from unittest.mock import MagicMock, patch

from interpreter.core.toolbox.web.web import Web, ResultItem, StructuredOutputResult, WebToolboxError

class TestWebToolbox(unittest.TestCase):
    def setUp(self):
        self.mock_toolbox = MagicMock()
        self.web = Web(self.mock_toolbox)

    def test_structured_output_linkup(self):
        """Verify dict schemas are JSON-encoded for the LinkUp SDK and results normalize correctly."""
        # Mock API key
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            # Mock LinkupClient
            with patch("linkup.LinkupClient") as MockClient:
                mock_instance = MockClient.return_value
                
                # Mock successful response
                mock_response = MagicMock()
                mock_response.structured_output = {
                    "author_last_name": "Vaswani",
                    "year": 2017,
                    "title": "Attention is All You Need"
                }
                mock_response.sources = [
                    {"title": "Paper on arXiv", "url": "https://arxiv.org/abs/1706.03762", "snippet": "We propose a new simple network architecture..."}
                ]
                mock_instance.search.return_value = mock_response
                
                # Define schema
                schema = {
                    "type": "object",
                    "properties": {
                        "author_last_name": {"type": "string"},
                        "year": {"type": "integer"},
                        "title": {"type": "string"}
                    }
                }
                
                # Call the method
                result = self.web.structured_output("Attention is All You Need", schema=schema)
                
                # Verify call parameters
                MockClient.assert_called_once_with(api_key="fake_key")
                mock_instance.search.assert_called_once()
                call_kwargs = mock_instance.search.call_args.kwargs
                self.assertEqual(call_kwargs["output_type"], "structured")
                # Backend receives JSON string for dict schemas
                self.assertEqual(call_kwargs["structured_output_schema"], json.dumps(schema))
                
                # Verify result structure
                self.assertIsInstance(result, StructuredOutputResult)
                self.assertEqual(result["structured_output"]["author_last_name"], "Vaswani")
                self.assertEqual(result["structured_output"]["year"], 2017)
                self.assertEqual(len(result["sources"]), 1)
                self.assertEqual(result["sources"][0]["title"], "Paper on arXiv")

    def test_structured_output_pydantic_flexibility(self):
        """Verify Pydantic model classes pass through to the LinkUp SDK unmodified."""
        # Mock API key
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            # Mock a Pydantic-like model by inheriting from a real one if available
            try:
                from pydantic import BaseModel
                class MockModel(BaseModel):
                    test: str
                original_schema = MockModel
            except ImportError:
                # Fallback to a mock that doesn't inherit but simulates the behavior
                class MockModel:
                    @staticmethod
                    def model_json_schema():
                        return {"type": "object", "properties": {"test": {"type": "string"}}}
                original_schema = MockModel
            
            with patch("linkup.LinkupClient") as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.search.return_value = MagicMock(structured_output={"test": "val"}, sources=[])
                
                # Call with pydantic-like object
                result = self.web.structured_output("query", schema=original_schema)
                
                # Verify call parameters - Linkup SDK receives the class itself
                call_kwargs = mock_instance.search.call_args.kwargs
                self.assertEqual(call_kwargs["structured_output_schema"], original_schema)

    def test_structured_output_no_backend_available(self):
        """Verify structured_output raises a helpful error when no API keys are configured."""
        # Ensure no API keys are set
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(WebToolboxError) as context:
                self.web.structured_output("test", schema={})
            # It might raise the specific ApiKeyError message or the aggregate No backends message
            err_msg = str(context.exception)
            self.assertTrue(
                "No structured output backends are working" in err_msg or 
                "LINKUP_API_KEY" in err_msg
            )

    def test_check_backend_available_vanshul_always_true(self):
        """Verify the keyless vanshul fetch backend reports available even with no env keys."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(self.web._check_backend_available("vanshul"))
            self.assertTrue(self.web._check_backend_available("VANSHUL"))

    def test_fetch_vanshul_success(self):
        """Verify fetch(backend='vanshul') returns normalized markdown content and title."""
        payloads = {
            "fetch_markdown": "Hello **world**",
            "fetch_metadata": {"title": "Example Domain"},
        }
        with patch.object(self.web, "_vanshul_mcp_call", side_effect=lambda name, args: payloads[name]):
            result = self.web.fetch("https://example.com", backend="vanshul")
            self.assertEqual(result["backend"], "vanshul")
            self.assertEqual(result["url"], "https://example.com")
            self.assertEqual(result["content"], "Hello **world**")
            self.assertEqual(result["title"], "Example Domain")

    def test_fetch_vanshul_metadata_failure_still_returns_content(self):
        """Verify a metadata failure degrades to an empty title instead of failing the fetch."""
        def fake_call(name, args):
            if name == "fetch_markdown":
                return "Some content"
            raise WebToolboxError("metadata failed")
        with patch.object(self.web, "_vanshul_mcp_call", side_effect=fake_call):
            result = self.web.fetch("https://example.com", backend="vanshul")
            self.assertEqual(result["content"], "Some content")
            self.assertEqual(result["title"], "")

    def test_fetch_vanshul_empty_content_raises(self):
        """Verify an empty markdown payload raises WebToolboxError so auto-select can fall through."""
        with patch.object(self.web, "_vanshul_mcp_call", return_value="   "):
            with self.assertRaises(WebToolboxError):
                self.web.fetch("https://example.com", backend="vanshul")

    def test_fetch_vanshul_rejects_multi_url(self):
        """Verify multi-URL fetch with the vanshul backend raises a tavily-only guidance error."""
        with self.assertRaises(WebToolboxError) as context:
            self.web.fetch(
                "https://example.com",
                backend="vanshul",
                urls=["https://example.com", "https://example.org"],
            )
        self.assertIn("tavily", str(context.exception))

    def test_search_page_vanshul_success(self):
        """Verify search_page(backend='vanshul') normalizes ranked passages and maps max_results."""
        payload = {
            "url": "https://example.com/",
            "query": "documentation",
            "count": 1,
            "matches": [{"heading": None, "snippet": "Use in documentation examples.", "score": 1}],
        }
        with patch.object(self.web, "_vanshul_mcp_call", return_value=payload) as mock_call:
            result = self.web.search_page("https://example.com", "documentation", backend="vanshul")
            self.assertEqual(result["backend"], "vanshul")
            self.assertEqual(len(result["matches"]), 1)
            self.assertEqual(result["matches"][0]["snippet"], "Use in documentation examples.")
            self.assertEqual(result["matches"][0]["score"], 1)
            posargs, _ = mock_call.call_args
            self.assertEqual(posargs[0], "search_page")
            self.assertEqual(posargs[1]["max_matches"], 5)

    def test_search_page_vanshul_empty_matches(self):
        """Verify zero matches is a valid empty result, not an error (term simply not on page)."""
        payload = {"url": "https://example.com/", "query": "xyzzy", "count": 0, "matches": []}
        with patch.object(self.web, "_vanshul_mcp_call", return_value=payload):
            result = self.web.search_page("https://example.com", "xyzzy", backend="vanshul")
            self.assertEqual(result["matches"], [])

    def test_search_page_tavily_passes_query(self):
        """Verify the tavily backend forwards query/chunks to extract and maps results to matches."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "fake_key"}):
            with patch("tavily.TavilyClient") as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.extract.return_value = {
                    "results": [
                        {"url": "https://example.com", "title": "Example", "content": "Rate limits apply.", "score": 0.9}
                    ],
                    "failed_results": [],
                }
                result = self.web.search_page("https://example.com", "rate limits", backend="tavily")
                call_kwargs = mock_instance.extract.call_args.kwargs
                self.assertEqual(call_kwargs["query"], "rate limits")
                self.assertEqual(call_kwargs["urls"], ["https://example.com"])
                self.assertEqual(result["backend"], "tavily")
                self.assertEqual(len(result["matches"]), 1)
                self.assertIn("Rate limits", result["matches"][0]["snippet"])

    def test_search_page_emulated_via_fetch_backend(self):
        """Verify backends without native support emulate search via full fetch plus local matching."""
        from interpreter.core.toolbox.web.web import FetchResult
        page = FetchResult({"url": "https://example.com", "title": "", "content": "Alpha pricing plans beta", "backend": "serper"})
        with patch.object(self.web, "fetch", return_value=page):
            result = self.web.search_page("https://example.com", "pricing", backend="serper")
            self.assertEqual(result["backend"], "serper")
            self.assertEqual(len(result["matches"]), 1)
            self.assertIn("pricing", result["matches"][0]["snippet"])
            self.assertIsNone(result["matches"][0]["score"])

    def test_search_page_invalid_backend(self):
        """Verify an unknown backend name raises a guidance error listing supported backends."""
        with self.assertRaises(WebToolboxError) as context:
            self.web.search_page("https://example.com", "x", backend="brave")
        self.assertIn("vanshul", str(context.exception))

    def test_search_page_auto_falls_through_to_fetch(self):
        """Verify auto-select falls back to fetch emulation and labels the fetch backend used."""
        from interpreter.core.toolbox.web.web import FetchResult
        page = FetchResult({"url": "https://example.com", "title": "", "content": "Hello world", "backend": "serper"})
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(self.web, "_search_page_vanshul", side_effect=WebToolboxError("down")):
                with patch.object(self.web, "fetch", return_value=page):
                    result = self.web.search_page("https://example.com", "hello")
                    # Vanshul raises, tavily has no key, so emulation via fetch() handles it,
                    # labeled with the fetch backend actually used (feedable back into backend=).
                    self.assertEqual(result["backend"], "serper")
                    self.assertEqual(result["raw_response"]["emulated_via_fetch"], "serper")
                    self.assertEqual(len(result["matches"]), 1)

    def test_page_search_result_fetch_returns_full_page(self):
        """Verify PageSearchResult.fetch() retrieves the full page the passages came from."""
        from interpreter.core.toolbox.web.web import FetchResult
        payload = {"url": "https://example.com/", "query": "q", "count": 0, "matches": []}
        full = FetchResult({"url": "https://example.com/", "title": "T", "content": "full", "backend": "vanshul"})
        with patch.object(self.web, "_vanshul_mcp_call", return_value=payload):
            with patch.object(self.web, "fetch", return_value=full) as mock_fetch:
                result = self.web.search_page("https://example.com", "q", backend="vanshul")
                self.assertEqual(result.fetch()["content"], "full")
                mock_fetch.assert_called_once_with("https://example.com/")

    def test_result_item_model_style_access(self):
        """Verify normalized items support the attribute access models kept guessing (r.title)."""
        item = self.web._normalize_result_item({"title": "T", "link": "http://x", "snippet": "S"})
        self.assertIsInstance(item, ResultItem)
        # Attribute access (the style that raised AttributeError on plain dicts)
        self.assertEqual(item.title, "T")
        self.assertEqual(item.url, "http://x")
        self.assertEqual(item.snippet, "S")
        # Key access still works
        self.assertEqual(item["title"], "T")
        self.assertEqual(item.get("url"), "http://x")

    def test_result_item_key_aliases(self):
        """Verify common key guesses resolve: content<->snippet, link/href->url, name->title."""
        item = self.web._normalize_result_item({"title": "T", "link": "http://x", "snippet": "S"})
        self.assertEqual(item["content"], "S")
        self.assertEqual(item.content, "S")
        linky = ResultItem({"title": "T", "link": "http://x", "description": "D"})
        self.assertEqual(linky.url, "http://x")
        self.assertEqual(linky["url"], "http://x")
        self.assertEqual(linky.snippet, "D")
        self.assertEqual(ResultItem({"name": "N", "url": "http://x", "snippet": "S"}).title, "N")
        # Reverse direction: fetch-style entries expose snippet as an alias of content
        page = ResultItem({"url": "http://x", "title": "T", "content": "C"})
        self.assertEqual(page.snippet, "C")
        self.assertEqual(page["snippet"], "C")
        self.assertEqual(page.get("snippet"), "C")

    def test_result_item_truly_missing_keys_still_error(self):
        """Verify forgiveness has limits: unknown keys raise KeyError/AttributeError, get() defaults."""
        item = ResultItem({"title": "T"})
        with self.assertRaises(KeyError):
            item["nope"]
        with self.assertRaises(AttributeError):
            item.nope
        self.assertIsNone(item.get("nope"))
        self.assertEqual(item.get("nope", "fallback"), "fallback")

    def test_search_page_matches_support_attribute_access(self):
        """Verify page-search passages support match.snippet as well as match['snippet']."""
        payload = {"url": "https://example.com/", "query": "q", "count": 1,
                   "matches": [{"heading": "H", "snippet": "S", "score": 2}]}
        with patch.object(self.web, "_vanshul_mcp_call", return_value=payload):
            result = self.web.search_page("https://example.com", "q", backend="vanshul")
            match = result.matches[0]
            self.assertIsInstance(match, ResultItem)
            self.assertEqual(match.snippet, "S")
            self.assertEqual(match.heading, "H")
            self.assertEqual(match["snippet"], "S")

if __name__ == "__main__":
    unittest.main()
