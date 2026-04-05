import os
import unittest
from unittest.mock import MagicMock, patch

from interpreter.core.toolbox.web.web import Web, StructuredOutputResult, WebToolboxError

class TestWebToolbox(unittest.TestCase):
    def setUp(self):
        self.mock_toolbox = MagicMock()
        self.web = Web(self.mock_toolbox)

    def test_structured_output_linkup(self):
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
                self.assertEqual(call_kwargs["structured_output_schema"], schema)
                
                # Verify result structure
                self.assertIsInstance(result, StructuredOutputResult)
                self.assertEqual(result["structured_output"]["author_last_name"], "Vaswani")
                self.assertEqual(result["structured_output"]["year"], 2017)
                self.assertEqual(len(result["sources"]), 1)
                self.assertEqual(result["sources"][0]["title"], "Paper on arXiv")

    def test_structured_output_pydantic_flexibility(self):
        # Mock API key
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            # Mock Pydantic-like object
            class MockModel:
                @staticmethod
                def model_json_schema():
                    return {"type": "object", "properties": {"test": {"type": "string"}}}
            
            with patch("linkup.LinkupClient") as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.search.return_value = MagicMock(structured_output={"test": "val"}, sources=[])
                
                # Call with pydantic-like object
                result = self.web.structured_output("query", schema=MockModel)
                
                # Verify call parameters - should have called model_json_schema()
                call_kwargs = mock_instance.search.call_args.kwargs
                self.assertEqual(call_kwargs["structured_output_schema"], {"type": "object", "properties": {"test": {"type": "string"}}})

    def test_structured_output_no_backend_available(self):
        # Ensure no API keys are set
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(WebToolboxError) as context:
                self.web.structured_output("test", schema={})
            self.assertIn("No structured output backends are working", str(context.exception))

if __name__ == "__main__":
    unittest.main()
