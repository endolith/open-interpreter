import os
import sys
import json
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from interpreter.core.toolbox.web.web import Web, StructuredOutputResult, WebToolboxError

@pytest.fixture
def web():
    mock_toolbox = MagicMock()
    return Web(mock_toolbox)

def test_structured_output_linkup(web):
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
            result = web.structured_output("Attention is All You Need", schema=schema)
            
            # Verify call parameters
            MockClient.assert_called_once_with(api_key="fake_key")
            mock_instance.search.assert_called_once()
            call_kwargs = mock_instance.search.call_args.kwargs
            assert call_kwargs["output_type"] == "structured"
            assert call_kwargs["structured_output_schema"] == schema
            
            # Verify result structure
            assert isinstance(result, StructuredOutputResult)
            assert result["structured_output"]["author_last_name"] == "Vaswani"
            assert result["structured_output"]["year"] == 2017
            assert len(result["sources"]) == 1
            assert result["sources"][0]["title"] == "Paper on arXiv"

def test_structured_output_pydantic_flexibility(web):
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
            result = web.structured_output("query", schema=MockModel)
            
            # Verify call parameters - should have called model_json_schema()
            call_kwargs = mock_instance.search.call_args.kwargs
            assert call_kwargs["structured_output_schema"] == {"type": "object", "properties": {"test": {"type": "string"}}}

def test_structured_output_no_backend_available(web):
    # Ensure no API keys are set
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(WebToolboxError) as excinfo:
            web.structured_output("test", schema={})
        assert "No structured output backends are working" in str(excinfo.value)
