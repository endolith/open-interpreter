"""Live provider errors, rendered for the user without OpenAI-specific advice.

A request with an invalid key fails before inference, so it costs nothing: this
matrix can check many providers cheaply. It asserts the two things #467 changed:

* the panel carries the **provider's own** message (the prefix naming the
  exception and provider survives the HTML/JSON unwrapping), and
* no OpenAI-only advice leaks into it, which the old handler printed for every
  provider.

Providers that need region/endpoint/credential setup (Azure, Vertex, Bedrock)
are excluded; they cannot be exercised with a dummy key and a bare model name.
OpenRouter is excluded because it has answered dummy keys for free-tier models,
so it produces a charge instead of an error.

Run with ``OI_RUN_PROVIDER_ERROR_TESTS=1`` (the dedicated CI job sets it).
"""

import os
from unittest import mock

import litellm
import pytest

from interpreter.core.respond import _error_body, _provider_hint, display_provider_error

pytestmark = [pytest.mark.integration, pytest.mark.provider_errors]

DUMMY_KEY = "sk-invalid-open-interpreter-provider-error-test"

# provider, model, the env var litellm reads if a key is not passed explicitly
PROVIDERS = [
    ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
    ("anthropic", "anthropic/claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY"),
    ("groq", "groq/llama-3.1-8b-instant", "GROQ_API_KEY"),
    ("deepseek", "deepseek/deepseek-chat", "DEEPSEEK_API_KEY"),
    ("mistral", "mistral/mistral-small-latest", "MISTRAL_API_KEY"),
    ("gemini", "gemini/gemini-2.0-flash", "GEMINI_API_KEY"),
    ("cohere", "cohere/command-r-plus", "COHERE_API_KEY"),
    ("together_ai", "together_ai/meta-llama/Llama-3-8b-chat-hf", "TOGETHERAI_API_KEY"),
    (
        "fireworks_ai",
        "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct",
        "FIREWORKS_AI_API_KEY",
    ),
    ("perplexity", "perplexity/sonar", "PERPLEXITYAI_API_KEY"),
    ("xai", "xai/grok-2-1212", "XAI_API_KEY"),
    ("cerebras", "cerebras/llama3.1-8b", "CEREBRAS_API_KEY"),
    ("sambanova", "sambanova/Meta-Llama-3.1-8B-Instruct", "SAMBANOVA_API_KEY"),
    ("nvidia_nim", "nvidia_nim/meta/llama3-8b-instruct", "NVIDIA_NIM_API_KEY"),
    ("ai21", "ai21/jamba-1.5-mini", "AI21_API_KEY"),
    (
        "hyperbolic",
        "hyperbolic/meta-llama/Meta-Llama-3-8B-Instruct",
        "HYPERBOLIC_API_KEY",
    ),
]

OPENAI_ONLY_ADVICE = (
    "OPENAI_API_KEY",
    "platform.openai.com",
    "purchase credits",
    "hosted `i` model",
)

_CONNECTION_FAILURE_MARKERS = (
    "connection error",
    "max retries",
    "timed out",
    "timeout",
    "name or service not known",
    "temporary failure in name resolution",
    "failed to establish a new connection",
)


class _Llm:
    api_key = DUMMY_KEY


class _Interpreter:
    llm = _Llm()


@pytest.mark.skipif(
    os.environ.get("OI_RUN_PROVIDER_ERROR_TESTS") != "1",
    reason="live provider error matrix; set OI_RUN_PROVIDER_ERROR_TESTS=1",
)
@pytest.mark.parametrize("provider,model,env_var", PROVIDERS, ids=[p[0] for p in PROVIDERS])
def test_a_dummy_key_error_shows_the_providers_own_message(
    monkeypatch, provider, model, env_var
):
    """A rejected dummy key renders the provider's text, never OpenAI advice.

    Every provider phrases a bad key differently — JSON, a plain sentence, an
    HTTP page — so this is the cross-provider check that #467's unwrapping
    keeps their wording instead of substituting ours.
    """
    monkeypatch.setenv(env_var, DUMMY_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    monkeypatch.setattr(litellm, "api_key", None, raising=False)

    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            api_key=DUMMY_KEY,
            max_tokens=1,
            timeout=20,
            num_retries=0,
        )
    except Exception as error:
        error_text = str(error)
        caught = error
    else:
        pytest.skip(f"{provider} accepted a dummy key; no error to render")

    lowered = error_text.lower()
    if isinstance(caught, litellm.exceptions.APIConnectionError) and any(
        marker in lowered for marker in _CONNECTION_FAILURE_MARKERS
    ):
        pytest.skip(f"{provider} unreachable from here: {error_text[:100]}")

    assert error_text.strip(), f"{provider} raised without a message"

    rendered = str(_error_body(error_text))
    assert rendered.strip()
    # The prefix names the exception class and provider; both unwrapping paths
    # keep it, so its first characters must survive.
    assert error_text.strip()[:30] in rendered

    for phrase in OPENAI_ONLY_ADVICE:
        assert phrase not in rendered, f"{provider}: OpenAI-only advice leaked in"

    with mock.patch("interpreter.core.respond.rich_print") as rich_print:
        display_provider_error(caught)
    # str(Panel) is the object repr; the user sees the Markdown it wraps.
    renderable = rich_print.call_args[0][0].renderable
    panel = str(getattr(renderable, "markup", renderable))
    assert error_text.strip()[:30] in panel
    for phrase in OPENAI_ONLY_ADVICE:
        assert phrase not in panel

    hint = _provider_hint(caught, _Interpreter()) or ""
    if isinstance(caught, litellm.exceptions.AuthenticationError):
        assert "Check the API key" in hint
    else:
        assert "openai" not in hint.lower()
        assert "hosted" not in hint.lower()
