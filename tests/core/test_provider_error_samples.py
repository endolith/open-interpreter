"""Captured provider error bodies, rendered exactly as the panel shows them.

These came from real providers with a rejected key or an unknown model: each
kept its own wording and shape (JSON, a plain sentence, an HTTP body). They are
the offline half of the live matrix in ``tests/test_provider_error_matrix.py``
and pin the rendering without network access.
"""

import pytest

from interpreter.core.respond import _error_body

OPENAI_ONLY_ADVICE = (
    "OPENAI_API_KEY",
    "platform.openai.com",
    "purchase credits",
    "hosted",
)

PROVIDER_ERROR_SAMPLES = {
    "anthropic": (
        'litellm.AuthenticationError: AnthropicException - {"type":"error",'
        '"error":{"type":"authentication_error","message":"invalid x-api-key"}}',
        "invalid x-api-key",
    ),
    "groq": (
        'litellm.BadRequestError: GroqException - {"error":{"message":"Invalid API Key",'
        '"type":"invalid_request_error","code":"invalid_api_key"}}',
        "Invalid API Key",
    ),
    "gemini": (
        'litellm.AuthenticationError: GeminiException - {\n  "error": {\n'
        '    "code": 400,\n    "message": "API key not valid. Please pass a valid API key.",\n'
        '    "status": "INVALID_ARGUMENT"\n  }\n}',
        "API key not valid",
    ),
    "openai-rate": (
        "litellm.RateLimitError: RateLimitError: OpenAIException - Your account is not "
        "active, please check your billing details on our website.",
        "Your account is not active",
    ),
    "openrouter-unknown-model": (
        'litellm.BadRequestError: OpenrouterException - {"error":{"message":'
        '"openai/definitely-not-a-real-model is not a valid model ID","code":400}}',
        "is not a valid model ID",
    ),
    "nvidia_nim": (
        "litellm.NotFoundError: Nvidia_nimException - 404 page not found",
        "404 page not found",
    ),
    "cohere": (
        'litellm.APIConnectionError: CohereException - {"id":"x",'
        '"message":"Incorrect API key provided"}',
        "Incorrect API key provided",
    ),
    "html-gateway": (
        "litellm.APIError: OpenrouterException - <!DOCTYPE html><html><body>"
        "<h1>502 Bad Gateway</h1></body></html>",
        "502 Bad Gateway",
    ),
}


@pytest.mark.parametrize(
    "error,expected", PROVIDER_ERROR_SAMPLES.values(), ids=PROVIDER_ERROR_SAMPLES.keys()
)
def test_provider_error_samples_keep_the_providers_words(error, expected):
    """Captured provider errors keep their message and gain no OpenAI advice.

    The old handler replaced whichever of these arrived with OpenAI key and
    billing instructions. Each sample must still contain the provider's own
    wording — including the HTML body, which must be unwrapped — and none of
    that OpenAI-only text.
    """
    rendered = str(_error_body(error))

    assert expected in rendered
    for phrase in OPENAI_ONLY_ADVICE:
        assert phrase not in rendered
    assert "<!DOCTYPE" not in rendered
    assert "<h1>" not in rendered
