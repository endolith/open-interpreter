"""Live provider errors, rendered for the user without OpenAI-specific advice.

A request that is rejected fails before inference, so it costs nothing: an
invalid key, a model name that does not exist, or a parameter outside the
accepted range. This matrix checks that the panel carries the **provider's own**
message — JSON body, plain sentence, or HTML page — and that no OpenAI-only
advice leaks into it, which the old handler printed for every provider.

Run with ``OI_RUN_PROVIDER_ERROR_TESTS=1`` (the dedicated CI job sets it).
"""

import json
import os
from unittest import mock

import litellm
import pytest

from interpreter.core.respond import _error_body, _provider_hint, display_provider_error

pytestmark = [pytest.mark.integration, pytest.mark.provider_errors]

DUMMY_KEY = "sk-invalid-open-interpreter-provider-error-test"

# provider, model, the env var litellm reads if a key is not passed explicitly.
# Deliberately keyless: these only need the rejection path.
DUMMY_KEY_PROVIDERS = [
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

# id, model, env var, extra request params, a substring only that error carries
# and the shape of the body it comes in. These need a real key, but every one is
# rejected before inference, so they are free. OpenRouter is first because it is
# the provider the integration CI keeps a working key for.
MODEL_AND_PARAMETER_ERRORS = [
    (
        "openrouter-unknown-model",
        "openrouter/openai/definitely-not-a-real-model",
        "OPENROUTER_API_KEY",
        {},
        "is not a valid model ID",
        "json",
    ),
    (
        "openrouter-temperature",
        "openrouter/openai/gpt-4o-mini",
        "OPENROUTER_API_KEY",
        {"temperature": 99},
        "Expected temperature to be at most 2",
        "json",
    ),
    (
        "openai-unknown-model",
        "openai/gpt-4oo",
        "OPENAI_API_KEY",
        {},
        "does not exist",
        "plain",
    ),
    (
        "anthropic-unknown-model",
        "anthropic/claude-not-a-real-model",
        "ANTHROPIC_API_KEY",
        {},
        "not_found",
        "json",
    ),
    (
        "deepseek-unknown-model",
        "deepseek/not-a-real-model",
        "DEEPSEEK_API_KEY",
        {},
        "supported API model names",
        "json",
    ),
    (
        "groq-unknown-model",
        "groq/not-a-real-model",
        "GROQ_API_KEY",
        {},
        "model",
        "json",
    ),
    (
        "mistral-unknown-model",
        "mistral/not-a-real-model",
        "MISTRAL_API_KEY",
        {},
        "model",
        "json",
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

_KEY_REJECTED_MARKERS = (
    "invalid api key",
    "incorrect api key",
    "invalid x-api-key",
    "authentication fails",
    "api key not valid",
)


class _Llm:
    api_key = DUMMY_KEY


class _Interpreter:
    llm = _Llm()


def _assert_rendered(caught, provider, expected=None, shape=None):
    """Assert the panel is exactly the provider's text, and the hint is neutral.

    Advice we add lives in the hint, not the panel: the panel is the provider's
    own words, and OpenAI's message itself points at
    https://platform.openai.com/account/api-keys. So the panel check is that it
    equals the rendered body (nothing appended), plus that a non-OpenAI provider
    did not somehow acquire OpenAI advice.
    """
    error_text = str(caught)
    assert error_text.strip(), f"{provider} raised without a message"

    rendered = str(_error_body(error_text))
    assert rendered.strip()
    # Both unwrapping paths keep the prefix naming the exception and provider.
    assert error_text.strip()[:30] in rendered

    if shape == "json":
        start, end = error_text.find("{"), error_text.rfind("}")
        payload = json.loads(error_text[start : end + 1])
        assert isinstance(payload, dict), f"{provider}: expected a JSON body"
    if expected:
        assert expected.lower() in rendered.lower(), (
            f"{provider}: {expected!r} missing from rendered body"
        )

    with mock.patch("interpreter.core.respond.rich_print") as rich_print:
        display_provider_error(caught)
    # str(Panel) is the object repr; the user sees the Markdown it wraps.
    renderable = rich_print.call_args[0][0].renderable
    panel = str(getattr(renderable, "markup", renderable))
    assert error_text.strip()[:30] in panel
    # Nothing is appended to the provider's text: the panel is the body.
    assert panel.strip() == rendered.strip()
    # OpenAI's own message names platform.openai.com; only a different provider
    # acquiring that text would mean our advice leaked into the panel.
    if provider != "openai":
        for phrase in OPENAI_ONLY_ADVICE:
            assert phrase not in panel, f"{provider}: OpenAI-only advice leaked in"

    hint = _provider_hint(caught, _Interpreter()) or ""
    for phrase in OPENAI_ONLY_ADVICE:
        assert phrase not in hint, f"{provider}: OpenAI-only advice in the hint"
    if isinstance(caught, litellm.exceptions.AuthenticationError):
        assert "Check the API key" in hint
    else:
        assert "openai" not in hint.lower()
        assert "hosted" not in hint.lower()


def _is_unreachable(caught, error_text):
    lowered = error_text.lower()
    return isinstance(caught, litellm.exceptions.APIConnectionError) and any(
        marker in lowered for marker in _CONNECTION_FAILURE_MARKERS
    )


def _is_key_rejected(caught, error_text):
    lowered = error_text.lower()
    return (
        isinstance(
            caught,
            (litellm.exceptions.AuthenticationError, litellm.exceptions.RateLimitError),
        )
        or any(marker in lowered for marker in _KEY_REJECTED_MARKERS)
    )


@pytest.mark.skipif(
    os.environ.get("OI_RUN_PROVIDER_ERROR_TESTS") != "1",
    reason="live provider error matrix; set OI_RUN_PROVIDER_ERROR_TESTS=1",
)
@pytest.mark.parametrize(
    "provider,model,env_var",
    DUMMY_KEY_PROVIDERS,
    ids=[p[0] for p in DUMMY_KEY_PROVIDERS],
)
def test_a_rejected_dummy_key_shows_the_providers_own_message(
    monkeypatch, provider, model, env_var
):
    """A rejected key renders the provider's text, never OpenAI advice.

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
        caught = error
        error_text = str(error)
    else:
        pytest.skip(f"{provider} accepted a dummy key; no error to render")

    if _is_unreachable(caught, error_text):
        pytest.skip(f"{provider} unreachable from here: {error_text[:100]}")

    _assert_rendered(caught, provider)


@pytest.mark.skipif(
    os.environ.get("OI_RUN_PROVIDER_ERROR_TESTS") != "1",
    reason="live provider error matrix; set OI_RUN_PROVIDER_ERROR_TESTS=1",
)
@pytest.mark.parametrize(
    "provider,model,env_var,params,expected,shape",
    MODEL_AND_PARAMETER_ERRORS,
    ids=[c[0] for c in MODEL_AND_PARAMETER_ERRORS],
)
def test_an_unknown_model_or_bad_parameter_shows_the_providers_reason(
    provider, model, env_var, params, expected, shape
):
    """A model or parameter the provider rejects reaches the user unaltered.

    This is the error type a user actually hits with a typo, and how the
    provider words it — JSON `message`, a plain sentence — is what tells them
    which field is wrong. Rejected before inference, so it costs nothing.
    """
    if not os.environ.get(env_var):
        pytest.skip(f"{env_var} not set; cannot reach {provider}'s validation")

    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=20,
            num_retries=0,
            **params,
        )
    except Exception as error:
        caught = error
        error_text = str(error)
    else:
        pytest.fail(f"{provider} accepted an invalid model/parameter: {model} {params}")

    if _is_unreachable(caught, error_text):
        pytest.skip(f"{provider} unreachable from here: {error_text[:100]}")
    if _is_key_rejected(caught, error_text):
        pytest.skip(f"{provider} key rejected; model/parameter error not reachable")

    _assert_rendered(caught, provider, expected=expected, shape=shape)
