import os
import unittest
import json
from unittest.mock import MagicMock, patch

from interpreter.core.toolbox.web.web import Web, StructuredOutputResult, WebToolboxError

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

    def test_vanshul_mcp_call_parses_jsonrpc_response(self):
        """Verify _vanshul_mcp_call unwraps the JSON-RPC envelope and JSON-parses text payloads."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": '{"title": "Hi"}'}]},
        }
        with patch("interpreter.core.toolbox.web.web.requests.post", return_value=mock_response):
            out = self.web._vanshul_mcp_call("fetch_metadata", {"url": "https://example.com"})
            self.assertEqual(out, {"title": "Hi"})

if __name__ == "__main__":
    unittest.main()
