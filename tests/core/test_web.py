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
                self.web.structured_output("test", schema={"name": "string"})
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
        with patch.object(self.web, "_vanshul_mcp_call", side_effect=lambda name, args, **kw: payloads[name]):
            result = self.web.fetch("https://example.com", backend="vanshul")
            self.assertEqual(result["backend"], "vanshul")
            self.assertEqual(result["url"], "https://example.com")
            self.assertEqual(result["content"], "Hello **world**")
            self.assertEqual(result["title"], "Example Domain")

    def test_fetch_vanshul_metadata_failure_still_returns_content(self):
        """Verify a metadata failure degrades to an empty title instead of failing the fetch."""
        def fake_call(name, args, **kw):
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

    def test_search_page_non_native_backend_guides(self):
        """Verify serper/linkup point at fetch().find() instead of emulating search."""
        for backend in ("serper", "linkup"):
            with self.subTest(backend=backend):
                with self.assertRaises(WebToolboxError) as context:
                    self.web.search_page("https://example.com", "pricing", backend=backend)
                msg = str(context.exception)
                self.assertIn("no native within-page search", msg)
                self.assertIn("page.find(term)", msg)

    def test_search_page_invalid_backend(self):
        """Verify an unknown backend name raises a guidance error listing supported backends."""
        with self.assertRaises(WebToolboxError) as context:
            self.web.search_page("https://example.com", "x", backend="brave")
        self.assertIn("vanshul", str(context.exception))

    def test_search_page_auto_falls_through_to_vanshul(self):
        """Verify auto-select falls from tavily failure through to keyless vanshul."""
        payload = {"url": "https://example.com/", "query": "hello", "count": 1,
                   "matches": [{"heading": None, "snippet": "Hello world", "score": 1}]}
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(self.web, "_search_page_tavily", side_effect=WebToolboxError("down")):
                with patch.object(self.web, "_vanshul_mcp_call", return_value=payload):
                    result = self.web.search_page("https://example.com", "hello")
                    self.assertEqual(result["backend"], "vanshul")
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

    def test_result_item_fetch_redirects(self):
        """Verify items built without a web handle still redirect to result methods."""
        item = ResultItem({"title": "T", "url": "http://a", "snippet": "S"})
        for method in ("fetch", "search_page"):
            with self.assertRaises(AttributeError) as context:
                getattr(item, method)
            msg = str(context.exception)
            self.assertIn(f"result.{method}(i)", msg)
            self.assertNotIn("item.title", msg)

    def test_item_bound_fetch_and_search_page(self):
        """Verify hits carry bound fetch()/search_page() over their own URL, data untouched."""
        from interpreter.core.toolbox.web.web import (
            AnswerResult, SearchResult, StructuredOutputResult,
        )
        web = MagicMock()
        web.fetch.return_value = "PAGE"
        web.search_page.return_value = "PASSAGES"
        hits = [{"title": "T", "url": "http://a", "snippet": "S"}]
        cases = [
            SearchResult({"results": hits, "backend": "x"}, web=web),
            AnswerResult({"answer": "a", "sources": hits, "backend": "x"}, web=web),
            StructuredOutputResult({"structured_output": {}, "sources": hits, "backend": "x"}, web=web),
        ]
        for result in cases:
            item = (result.get("results") or result.get("sources"))[0]
            with self.subTest(cls=type(result).__name__):
                self.assertEqual(item.fetch(), "PAGE")
                web.fetch.assert_called_with("http://a")
                self.assertEqual(item.search_page("q", max_results=2), "PASSAGES")
                web.search_page.assert_called_with("http://a", "q", max_results=2)
                self.assertNotIn("fetch", item.keys())
                self.assertNotIn("search_page", item.keys())

    def test_find_links_redirect_to_pages(self):
        """Verify result.find()/links() point at page methods instead of generic guidance."""
        from interpreter.core.toolbox.web.web import (
            AnswerResult, PageSearchResult, SearchResult, StructuredOutputResult,
        )
        cases = [
            (SearchResult({"results": [], "backend": "x"}, web=self.web), "result.fetch(i)"),
            (AnswerResult({"answer": "a", "sources": [], "backend": "x"}, web=self.web), "result.fetch(i)"),
            (StructuredOutputResult({"structured_output": {}, "sources": [], "backend": "x"}, web=self.web), "result.fetch(i)"),
            (PageSearchResult({"url": "http://a", "query": "q", "matches": [], "backend": "x"}, web=self.web), "result.fetch()"),
        ]
        for obj, hint in cases:
            for method in ("find", "links"):
                with self.subTest(cls=type(obj).__name__, method=method):
                    with self.assertRaises(AttributeError) as context:
                        getattr(obj, method)
                    msg = str(context.exception)
                    self.assertIn(hint, msg)
                    self.assertIn("search_page(i, query)", msg)

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

    def test_structured_output_simple_field_map_converts(self):
        """Verify a simple field map becomes a full object schema with all fields required."""
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with patch("linkup.LinkupClient") as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.search.return_value = MagicMock(structured_output={}, sources=[])
                self.web.structured_output("Apple Inc", schema={"name": "string", "founded": "integer"})
                sent = mock_instance.search.call_args.kwargs["structured_output_schema"]
                self.assertEqual(json.loads(sent), {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "founded": {"type": "integer"}},
                    "required": ["name", "founded"],
                })

    def test_structured_output_simple_map_invalid_type(self):
        """Verify an unknown type name fails fast locally instead of as a backend 400."""
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with patch("linkup.LinkupClient") as MockClient:
                with self.assertRaises(WebToolboxError) as context:
                    self.web.structured_output("q", schema={"name": "str"})
                self.assertIn("Valid types", str(context.exception))
                MockClient.assert_not_called()

    def test_structured_output_field_named_type(self):
        """Verify the ambiguity resolves to a field: {'type': 'string'} means a field named type."""
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with patch("linkup.LinkupClient") as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.search.return_value = MagicMock(structured_output={}, sources=[])
                self.web.structured_output("q", schema={"type": "string"})
                sent = mock_instance.search.call_args.kwargs["structured_output_schema"]
                self.assertEqual(json.loads(sent)["properties"], {"type": {"type": "string"}})

    def test_structured_output_mixed_property_schema(self):
        """Verify property-schema dicts pass through inside an otherwise simple field map."""
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with patch("linkup.LinkupClient") as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.search.return_value = MagicMock(structured_output={}, sources=[])
                self.web.structured_output("q", schema={"tags": {"type": "array", "items": {"type": "string"}}})
                sent = mock_instance.search.call_args.kwargs["structured_output_schema"]
                self.assertEqual(
                    json.loads(sent)["properties"]["tags"],
                    {"type": "array", "items": {"type": "string"}},
                )

    def test_structured_output_empty_schema_raises(self):
        """Verify an empty schema raises a helpful error instead of a backend 400."""
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with self.assertRaises(WebToolboxError) as context:
                self.web.structured_output("q", schema={})
            self.assertIn("at least one field", str(context.exception))

    def test_structured_output_schema_rejection_hint(self):
        """Verify a backend schema 400 blames the schema, not the API key."""
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with patch("linkup.LinkupClient") as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.search.side_effect = Exception(
                    "Validation failed structuredOutputSchema: must be valid JSON schema of type object."
                )
                with self.assertRaises(WebToolboxError) as context:
                    self.web.structured_output("q", schema={"name": "string"})
                msg = str(context.exception)
                self.assertIn("schema", msg.lower())
                self.assertNotIn("API key", msg)

    def test_search_result_search_page_delegates(self):
        """Verify SearchResult.search_page(i, query) searches within that result's URL."""
        from interpreter.core.toolbox.web.web import SearchResult
        result = SearchResult(
            {"results": [{"title": "T", "url": "http://a", "snippet": "S"}], "backend": "serper"},
            web=self.web,
        )
        with patch.object(self.web, "search_page", return_value="PASSAGES") as mock_sp:
            self.assertEqual(result.search_page(0, "pricing", max_results=3), "PASSAGES")
            mock_sp.assert_called_once_with("http://a", "pricing", max_results=3)

    def test_answer_result_search_page_delegates(self):
        """Verify AnswerResult.search_page(i, query) searches within that source's URL."""
        from interpreter.core.toolbox.web.web import AnswerResult
        result = AnswerResult(
            {"answer": "A", "sources": [{"title": "T", "url": "http://b", "snippet": "S"}], "backend": "linkup"},
            web=self.web,
        )
        with patch.object(self.web, "search_page", return_value="PASSAGES") as mock_sp:
            self.assertEqual(result.search_page(0, "query"), "PASSAGES")
            mock_sp.assert_called_once_with("http://b", "query")

    def test_structured_result_search_page_no_sources(self):
        """Verify StructuredOutputResult.search_page raises helpfully when there are no sources."""
        from interpreter.core.toolbox.web.web import StructuredOutputResult
        result = StructuredOutputResult({"structured_output": {}, "sources": [], "backend": "linkup"}, web=self.web)
        with self.assertRaises(WebToolboxError):
            result.search_page(0, "query")

    def test_fetch_auto_tries_vanshul_first(self):
        """Verify fetch auto-select tries the leanest keyless backend before keyed ones."""
        from interpreter.core.toolbox.web.web import FetchResult
        with patch.dict(os.environ, {"SERPER_API_KEY": "fake"}, clear=True):
            with patch.object(self.web, "_fetch_vanshul", side_effect=WebToolboxError("down")) as mock_v:
                made = FetchResult({"url": "https://example.com", "title": "", "content": "hi", "backend": "serper"})
                with patch.object(self.web, "_fetch_serper", return_value=dict(made)) as mock_s:
                    with patch.object(self.web, "_fetch_linkup") as mock_l:
                        with patch.object(self.web, "_fetch_tavily") as mock_t:
                            result = self.web.fetch("https://example.com")
                            self.assertEqual(result["backend"], "serper")
                            mock_v.assert_called_once()
                            mock_s.assert_called_once()
                            mock_l.assert_not_called()
                            mock_t.assert_not_called()


    def test_search_page_auto_prefers_tavily(self):
        """Verify search_page auto-select prefers semantic (tavily) over keyword (vanshul)."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "fake"}, clear=True):
            with patch.object(self.web, "_search_page_tavily", return_value={
                "url": "https://example.com", "query": "q", "matches": [], "raw_response": {},
            }) as mock_t:
                with patch.object(self.web, "_search_page_vanshul") as mock_v:
                    result = self.web.search_page("https://example.com", "paraphrased query")
                    self.assertEqual(result["backend"], "tavily")
                    mock_t.assert_called_once()
                    mock_v.assert_not_called()


    def test_fetch_prepends_missing_scheme(self):
        """Verify schemeless URLs gain https:// before reaching any backend."""
        from interpreter.core.toolbox.web.web import FetchResult
        page = FetchResult({"url": "https://example.com", "title": "", "content": "hi", "backend": "vanshul"})
        with patch.object(self.web, "_fetch_vanshul", return_value=dict(page)) as mock_fetch:
            result = self.web.fetch("example.com", backend="vanshul")
            self.assertEqual(result["url"], "https://example.com")
            mock_fetch.assert_called_once_with("https://example.com", timeout=30)


    def test_fetch_rejects_malformed_urls(self):
        """Verify malformed URLs raise helpfully instead of reaching backends."""
        for bad in ["not a url", "", "ftp://example.com/files", "https://", "http://"]:
            with self.subTest(url=bad):
                with self.assertRaises(WebToolboxError) as context:
                    self.web.fetch(bad, backend="vanshul")
                self.assertIn("https://example.com", str(context.exception))

    def test_fetch_integer_suggests_result_methods(self):
        """Verify an index passed as URL points at result.fetch(i)/search_page(i)."""
        with self.assertRaises(WebToolboxError) as context:
            self.web.fetch(1, backend="vanshul")
        msg = str(context.exception)
        self.assertIn("result.fetch(1)", msg)
        self.assertIn("not indices", msg)
        with self.assertRaises(WebToolboxError) as context:
            self.web.search_page(2, "q", backend="vanshul")
        self.assertIn("result.search_page(2", str(context.exception))


    def test_fetch_cache_shared_across_scheme_forms(self):
        """Verify schemeless and https:// forms of a URL share one cache entry."""
        from interpreter.core.toolbox.web.web import FetchResult
        made = FetchResult({"url": "https://example.com", "title": "", "content": "hi", "backend": "vanshul"})
        with patch.object(self.web, "_fetch_vanshul", return_value=dict(made)) as mock_fetch:
            first = self.web.fetch("example.com")
            second = self.web.fetch("https://example.com")
            self.assertEqual(first["content"], "hi")
            self.assertTrue(second._cached)
            self.assertEqual(mock_fetch.call_count, 1)


    def test_search_page_validates_url(self):
        """Verify search_page rejects malformed URLs and prepends a missing scheme."""
        with self.assertRaises(WebToolboxError):
            self.web.search_page("not a url", "q", backend="vanshul")
        payload = {"url": "https://example.com/", "query": "q", "count": 0, "matches": []}
        with patch.object(self.web, "_vanshul_mcp_call", return_value=payload) as mock_call:
            self.web.search_page("example.com", "q", backend="vanshul")
            posargs, _ = mock_call.call_args
            self.assertEqual(posargs[1]["url"], "https://example.com")


    def test_fetch_vanshul_error_text_raises(self):
        """Verify service error text (e.g. 'Error: Upstream returned 530') raises, never content."""
        with patch.object(self.web, "_vanshul_mcp_call", return_value="Error: Upstream returned 530"):
            with self.assertRaises(WebToolboxError) as context:
                self.web.fetch("https://example.com", backend="vanshul")
            self.assertIn("Vanshul could not fetch", str(context.exception))


    def test_search_page_vanshul_error_text_raises(self):
        """Verify service error text from search_page raises instead of shape errors."""
        with patch.object(self.web, "_vanshul_mcp_call", return_value="Error: Upstream returned 530"):
            with self.assertRaises(WebToolboxError) as context:
                self.web.search_page("https://example.com", "q", backend="vanshul")
            self.assertIn("Vanshul could not search", str(context.exception))


    def test_links_inline_title_and_parens(self):
        """Verify links() strips optional titles and keeps balanced parens in URLs."""
        from interpreter.core.toolbox.web.web import FetchResult
        page = FetchResult({
            "url": "https://en.wikipedia.org/wiki/X",
            "title": "",
            "content": '[Python](https://en.wikipedia.org/wiki/Python_(programming_language) "Python") and [A](http://a)',
            "backend": "tavily",
        })
        self.assertEqual(page.links(), [
            ("Python", "https://en.wikipedia.org/wiki/Python_(programming_language)"),
            ("A", "http://a"),
        ])


    def test_links_reference_style(self):
        """Verify links() resolves [text][ref] via [ref]: url definitions."""
        from interpreter.core.toolbox.web.web import FetchResult
        page = FetchResult({
            "url": "https://example.com",
            "title": "",
            "content": "See [docs][d] and [home][]\n\n[d]: https://example.com/docs\n[home]: https://example.com/",
            "backend": "serper",
        })
        self.assertEqual(page.links(), [
            ("docs", "https://example.com/docs"),
            ("home", "https://example.com/"),
        ])


    def test_links_reference_uses_without_definitions(self):
        """Verify dangling [text][ref] uses (serper strips definitions) yield no phantom links."""
        from interpreter.core.toolbox.web.web import FetchResult
        page = FetchResult({
            "url": "https://example.com",
            "title": "",
            "content": "See [docs][21] for details.",
            "backend": "serper",
        })
        self.assertEqual(page.links(), [])


    def test_fetch_linkup_old_sdk_guard(self):
        """Verify a linkup-sdk without fetch support raises an upgrade hint, not an auth error."""
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with patch("linkup.LinkupClient") as MockClient:
                del MockClient.return_value.fetch
                with self.assertRaises(WebToolboxError) as context:
                    self.web.fetch("https://example.com", backend="linkup")
                msg = str(context.exception)
                self.assertIn("--upgrade linkup-sdk", msg)
                self.assertNotIn("API key", msg)

    def test_fetch_list_index_guides(self):
        """Verify fetch([0, 1]) fails with guidance toward one-at-a-time fetching."""
        from interpreter.core.toolbox.web.web import SearchResult
        result = SearchResult(
            {"results": [{"title": "T", "url": "http://a", "snippet": "S"}], "backend": "serper"},
            web=self.web,
        )
        with self.assertRaises(WebToolboxError) as context:
            result.fetch([0, 1])
        msg = str(context.exception)
        self.assertIn("single result index", msg)
        self.assertIn("search_page(i, query)", msg)

    def test_fetch_string_index_guides(self):
        """Verify a non-integer index raises WebToolboxError, not TypeError."""
        from interpreter.core.toolbox.web.web import AnswerResult
        result = AnswerResult(
            {"answer": "A", "sources": [{"title": "T", "url": "http://b", "snippet": "S"}], "backend": "linkup"},
            web=self.web,
        )
        with self.assertRaises(WebToolboxError):
            result.fetch("0")
        with self.assertRaises(WebToolboxError):
            result.search_page({"i": 0}, "q")

    def test_fetch_out_of_range_guides(self):
        """Verify an out-of-range index names the valid range instead of leaking IndexError."""
        from interpreter.core.toolbox.web.web import StructuredOutputResult
        result = StructuredOutputResult(
            {"structured_output": {}, "sources": [{"title": "T", "url": "http://c", "snippet": "S"}], "backend": "linkup"},
            web=self.web,
        )
        with self.assertRaises(WebToolboxError) as context:
            result.fetch(5)
        self.assertIn("out of range", str(context.exception))

    def test_fetch_bool_index_rejected(self):
        """Verify bool is not silently accepted as an integer index."""
        from interpreter.core.toolbox.web.web import SearchResult
        result = SearchResult(
            {"results": [{"title": "T", "url": "http://a", "snippet": "S"}], "backend": "serper"},
            web=self.web,
        )
        with self.assertRaises(WebToolboxError):
            result.fetch(True)

    def test_search_repr_steers_to_methods(self):
        """Verify the repr funnels agents into fetch(i)/search_page(i) without pasting URLs."""
        from interpreter.core.toolbox.web.web import SearchResult
        result = SearchResult(
            {"results": [
                {"title": "T1", "url": "https://example.com/a/b?c=d", "snippet": "S1"},
                {"title": "T2", "url": "https://example.org/e", "snippet": "S2"},
            ], "backend": "serper"},
            web=self.web,
        )
        text = repr(result)
        # Method funnel advertised; domains shown for orientation...
        self.assertIn("hit.fetch()", text)
        self.assertIn("hit.search_page(query)", text)
        self.assertIn("NEVER invent hardcoded URLs", text)
        self.assertIn("example.com", text)
        # ...but full URLs are withheld so agents use methods instead of copying.
        self.assertNotIn("https://example.com/a/b?c=d", text)
        self.assertNotIn("https://example.org/e", text)

    def test_search_page_total_failure_reports_reasons(self):
        """Verify total failure surfaces each native backend's reason."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "fake"}, clear=True):
            with patch.object(self.web, "_search_page_tavily", side_effect=WebToolboxError("tavily down")):
                with patch.object(self.web, "_search_page_vanshul", side_effect=WebToolboxError("vanshul down")):
                    with self.assertRaises(WebToolboxError) as context:
                        self.web.search_page("https://example.com", "q")
                    msg = str(context.exception)
                    self.assertIn("tavily down", msg)
                    self.assertIn("vanshul down", msg)

    def test_search_page_no_backends_names_keys(self):
        """Verify total unavailability names the real keys that would unlock backends."""
        with patch.object(self.web, "_check_backend_available", return_value=False):
            with self.assertRaises(WebToolboxError) as context:
                self.web.search_page("https://example.com", "q")
            self.assertIn("TAVILY_API_KEY", str(context.exception))

    def test_run_with_timeout_fires(self):
        """Verify the thread guard returns fast values, propagates errors, and times out stalls."""
        import time
        from interpreter.core.toolbox.web.web import _run_with_timeout
        self.assertEqual(_run_with_timeout(lambda: 42, 5, "Test"), 42)
        with self.assertRaises(ValueError):
            _run_with_timeout(lambda: (_ for _ in ()).throw(ValueError("boom")), 5, "Test")
        started = time.time()
        with self.assertRaises(WebToolboxError) as context:
            _run_with_timeout(lambda: time.sleep(30), 0.3, "TestBackend")
        self.assertLess(time.time() - started, 10)
        self.assertIn("timed out", str(context.exception))

    def test_linkup_search_timeout_guard(self):
        """Verify a stalled linkup SDK call fails fast instead of hanging forever."""
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with patch("linkup.LinkupClient") as MockClient:
                import time
                mock_instance = MockClient.return_value
                mock_instance.search.side_effect = lambda **kw: time.sleep(30)
                started = time.time()
                with self.assertRaises(WebToolboxError) as context:
                    self.web.search("q", backend="linkup", timeout=0.3)
                self.assertLess(time.time() - started, 10)
                self.assertIn("timed out", str(context.exception))

    def test_serpapi_search_timeout_guard(self):
        """Verify a stalled serpapi call fails fast despite the SDK's 60000s default."""
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "fake_key"}):
            with patch("serpapi.GoogleSearch") as MockSearch:
                import time
                mock_instance = MockSearch.return_value
                mock_instance.get_dict.side_effect = lambda: time.sleep(30)
                started = time.time()
                with self.assertRaises(WebToolboxError) as context:
                    self.web.search("q", backend="serpapi", timeout=0.3)
                self.assertLess(time.time() - started, 10)
                self.assertIn("timed out", str(context.exception))

    def test_timeout_threaded_to_requests_backend(self):
        """Verify the public timeout reaches the requests call and defaults to 30."""
        from interpreter.core.toolbox.web.web import DEFAULT_TIMEOUT
        self.assertEqual(DEFAULT_TIMEOUT, 30)
        with patch.dict(os.environ, {"SERPER_API_KEY": "fake_key"}):
            with patch("interpreter.core.toolbox.web.web.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"organic": []}
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response
                self.web.fetch("https://example.com", backend="serper", timeout=7)
                self.assertEqual(mock_post.call_args.kwargs["timeout"], 7)

    def test_timeout_hidden_from_signatures(self):
        """Verify timeout costs models zero tokens: absent from all public signatures."""
        import inspect
        for method in ("search", "answer", "structured_output", "fetch", "search_page"):
            with self.subTest(method=method):
                self.assertNotIn("timeout", inspect.signature(getattr(self.web, method)).parameters)

    def test_timeout_from_profile_setting(self):
        """Verify the toolbox profile value drives the backend wait without a parameter."""
        from interpreter.core.toolbox.web.web import FetchResult
        made = FetchResult({"url": "https://example.com", "title": "", "content": "hi", "backend": "vanshul"})
        self.web.toolbox.web_timeout = 7
        try:
            with patch.object(self.web, "_fetch_vanshul", return_value=dict(made)) as mock_fetch:
                self.web.fetch("https://example.com")
                self.assertEqual(mock_fetch.call_args.kwargs["timeout"], 7)
        finally:
            del self.web.toolbox.web_timeout

    def test_timeout_invalid_profile_falls_back(self):
        """Verify garbage profile values degrade to defaults instead of crashing."""
        from interpreter.core.toolbox.web.web import DEFAULT_ANSWER_TIMEOUT, DEFAULT_TIMEOUT
        self.web.toolbox.web_timeout = "soon"
        self.web.toolbox.web_answer_timeout = None
        try:
            self.assertEqual(self.web._web_timeout(), DEFAULT_TIMEOUT)
            self.assertEqual(self.web._web_timeout("answer"), DEFAULT_ANSWER_TIMEOUT)
        finally:
            del self.web.toolbox.web_timeout
            del self.web.toolbox.web_answer_timeout

    def test_answer_default_timeout_is_generous(self):
        """Verify synthesis waits longer by default than plain fetches."""
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with patch.object(self.web, "_answer_linkup", return_value={"answer": "a", "sources": []}) as mock_a:
                self.web.answer("What is the answer?")
                self.assertEqual(mock_a.call_args.kwargs["timeout"], 120)

if __name__ == "__main__":
    unittest.main()
