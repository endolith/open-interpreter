"""Vision capability detection for provider-prefixed models."""

import pytest

from interpreter.core.llm import llm as llm_module
from interpreter.core.llm import providers


def test_litellm_positive_short_circuits_catalog(monkeypatch):
    """A model litellm already knows must not trigger a catalog fetch."""
    monkeypatch.setattr(llm_module.litellm, "supports_vision", lambda model: True)

    def fail(model):
        raise AssertionError("catalog must not be consulted")

    monkeypatch.setattr(llm_module, "openrouter_model_entry", fail)
    assert llm_module._supports_vision("openrouter/openai/gpt-4o-mini") is True


def test_catalog_marks_openrouter_model_vision_capable(monkeypatch):
    """The exact slug's input modalities decide for an OpenRouter model."""
    monkeypatch.setattr(llm_module.litellm, "supports_vision", lambda model: False)
    monkeypatch.setattr(
        llm_module,
        "openrouter_model_entry",
        lambda model: {"architecture": {"input_modalities": ["text", "image"]}},
    )
    assert llm_module._supports_vision("openrouter/openai/gpt-4o-mini") is True


def test_catalog_text_only_model_is_not_vision_capable(monkeypatch):
    """A catalog entry without the image modality stays text-only."""
    monkeypatch.setattr(llm_module.litellm, "supports_vision", lambda model: False)
    monkeypatch.setattr(
        llm_module,
        "openrouter_model_entry",
        lambda model: {"architecture": {"input_modalities": ["text"]}},
    )
    assert llm_module._supports_vision("openrouter/vendor/text-only") is False


def test_absent_catalog_entry_is_conservative(monkeypatch):
    """An unknown OpenRouter slug must not be assumed to accept images."""
    monkeypatch.setattr(llm_module.litellm, "supports_vision", lambda model: False)
    monkeypatch.setattr(llm_module, "openrouter_model_entry", lambda model: None)
    assert llm_module._supports_vision("openrouter/vendor/unknown") is False


def test_non_openrouter_model_never_queries_catalog(monkeypatch):
    """Only OpenRouter slugs may consult the OpenRouter catalog."""
    monkeypatch.setattr(llm_module.litellm, "supports_vision", lambda model: False)

    def fail(model):
        raise AssertionError("catalog must not be consulted for other providers")

    monkeypatch.setattr(llm_module, "openrouter_model_entry", fail)
    assert llm_module._supports_vision("gpt-3.5-turbo") is False


class _Response:
    """Minimal stand-in for a requests response."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_failure_is_not_cached(monkeypatch):
    """A transient fetch failure must not disable the model for the process."""
    calls = {"count": 0}

    def get(url, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("network down")
        return _Response(
            {
                "data": [
                    {
                        "id": "openai/gpt-4o-mini",
                        "architecture": {"input_modalities": ["text", "image"]},
                    }
                ]
            }
        )

    monkeypatch.setattr(providers.requests, "get", get)
    monkeypatch.setattr(providers, "_openrouter_models", None)

    with pytest.raises(providers.OpenRouterCatalogError):
        providers.openrouter_model_entry("openrouter/openai/gpt-4o-mini")
    entry = providers.openrouter_model_entry("openrouter/openai/gpt-4o-mini")
    assert entry["id"] == "openai/gpt-4o-mini"
    assert calls["count"] == 2


def test_openrouter_catalog_headers_default_to_this_fork(monkeypatch):
    """OpenRouter attribution defaults to this fork, not an empty referer."""
    captured = {}

    def get(url, **kwargs):
        captured.update(kwargs.get("headers", {}))
        return _Response({"data": []})

    monkeypatch.delenv("OR_SITE_URL", raising=False)
    monkeypatch.delenv("OR_APP_NAME", raising=False)
    monkeypatch.setattr(providers.requests, "get", get)
    monkeypatch.setattr(providers, "_openrouter_models", None)

    providers.openrouter_model_entry("openrouter/openai/gpt-4o-mini")

    assert captured["HTTP-Referer"] == "https://github.com/endolith/open-interpreter"
    assert captured["X-Title"] == "Open Interpreter"


def test_other_provider_makes_no_request(monkeypatch):
    """A non-OpenRouter id returns before any HTTP call."""
    def fail(url, **kwargs):
        raise AssertionError("no request expected")

    monkeypatch.setattr(providers.requests, "get", fail)
    monkeypatch.setattr(providers, "_openrouter_models", None)
    assert providers.openrouter_model_entry("openai/gpt-4o-mini") is None


def test_successful_fetch_is_cached(monkeypatch):
    """The catalog is fetched once and reused for every later slug."""
    calls = {"count": 0}

    def get(url, **kwargs):
        calls["count"] += 1
        return _Response(
            {
                "data": [
                    {"id": "vendor/one", "architecture": {"input_modalities": ["text"]}},
                    {
                        "id": "vendor/two",
                        "architecture": {"input_modalities": ["text", "image"]},
                    },
                ]
            }
        )

    monkeypatch.setattr(providers.requests, "get", get)
    monkeypatch.setattr(providers, "_openrouter_models", None)

    assert providers.openrouter_model_entry("openrouter/vendor/one")["id"] == "vendor/one"
    assert providers.openrouter_model_entry("openrouter/vendor/two")["id"] == "vendor/two"
    assert calls["count"] == 1


def test_http_error_is_not_cached(monkeypatch):
    """An HTTP error behaves like a fetch failure and is retried next time."""
    calls = {"count": 0}

    class _BadResponse(_Response):
        def raise_for_status(self):
            raise RuntimeError("500")

    def get(url, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _BadResponse({"data": []})
        return _Response(
            {"data": [{"id": "vendor/one", "architecture": {"input_modalities": ["image"]}}]}
        )

    monkeypatch.setattr(providers.requests, "get", get)
    monkeypatch.setattr(providers, "_openrouter_models", None)

    with pytest.raises(providers.OpenRouterCatalogError):
        providers.openrouter_model_entry("openrouter/vendor/one")
    assert providers.openrouter_model_entry("openrouter/vendor/one")["id"] == "vendor/one"
    assert calls["count"] == 2


def test_entry_without_modalities_is_conservative(monkeypatch):
    """An entry missing architecture or input_modalities is not vision-capable."""
    monkeypatch.setattr(llm_module.litellm, "supports_vision", lambda model: False)
    monkeypatch.setattr(llm_module, "openrouter_model_entry", lambda model: {"id": "vendor/one"})
    assert llm_module._supports_vision("openrouter/vendor/one") is False


def test_response_without_data_is_a_failure(monkeypatch):
    """A 200 without the model list must not cache an empty catalog."""
    monkeypatch.setattr(
        providers.requests, "get", lambda url, **kwargs: _Response({"object": "list"})
    )
    monkeypatch.setattr(providers, "_openrouter_models", None)

    with pytest.raises(providers.OpenRouterCatalogError):
        providers.openrouter_model_entry("openrouter/vendor/one")
    assert providers._openrouter_models is None


def test_catalog_failure_leaves_detection_unknown(monkeypatch):
    """A fetch failure must keep retrying instead of pinning text-only."""
    monkeypatch.setattr(llm_module.litellm, "supports_vision", lambda model: False)

    def fail(model):
        raise providers.OpenRouterCatalogError("down")

    monkeypatch.setattr(llm_module, "openrouter_model_entry", fail)
    assert llm_module._supports_vision("openrouter/vendor/one") is None


def test_entry_without_id_is_skipped(monkeypatch):
    """Malformed catalog entries do not crash the lookup."""
    monkeypatch.setattr(
        providers.requests,
        "get",
        lambda url, **kwargs: _Response(
            {"data": [{"architecture": {}}, {"id": "vendor/one", "architecture": {}}]}
        ),
    )
    monkeypatch.setattr(providers, "_openrouter_models", None)

    assert providers.openrouter_model_entry("openrouter/vendor/one")["id"] == "vendor/one"
    assert providers.openrouter_model_entry("openrouter/vendor/missing") is None


class _Vision:
    """Stub vision renderer needed to construct an Llm."""

    def query(self, lmc):
        return "a description"


class _Computer:
    vision = _Vision()


class _Interpreter:
    computer = _Computer()
    os = False
    shrink_images = False
    verbose = False
    in_terminal_interface = False
    import_computer_api = False
    always_apply_user_message_template = False
    user_message_template = "{content}"


def test_run_aborts_without_sending_when_catalog_is_unavailable(monkeypatch):
    """An unresolved vision capability must not send the request at all.

    Proceeding with supports_vision=None would skip both the description
    renderer and conversion's ``vision == False`` guard, sending raw image
    data to a model whose capability is unknown.
    """
    llm = llm_module.Llm(_Interpreter())
    llm.model = "openrouter/vendor/unknown"
    llm._is_loaded = True
    llm.supports_functions = True

    sent = []
    llm.completions = lambda **kwargs: sent.append(kwargs)

    monkeypatch.setattr(llm_module.litellm, "supports_vision", lambda model: False)

    def fail(model):
        raise providers.OpenRouterCatalogError("catalog down")

    monkeypatch.setattr(llm_module, "openrouter_model_entry", fail)

    messages = [
        {"role": "system", "type": "message", "content": "system"},
        {"role": "user", "type": "image", "format": "path", "content": "/tmp/x.png"},
    ]

    with pytest.raises(providers.OpenRouterCatalogError) as excinfo:
        list(llm.run(messages))

    assert "openrouter/vendor/unknown" in str(excinfo.value)
    assert "supports_vision" in str(excinfo.value)
    assert sent == []
    assert llm.supports_vision is None


def _llm_for_run():
    """An Llm ready for run() without loading or trimming by model lookup."""
    llm = llm_module.Llm(_Interpreter())
    llm.model = "openrouter/vendor/unknown"
    llm._is_loaded = True
    llm.supports_functions = True
    llm.context_window = 8000
    llm.max_tokens = 1000
    return llm


def test_run_fails_before_a_text_only_turn_when_catalog_is_unavailable(monkeypatch):
    """A text-only turn must fail up front, not run until an image appears."""
    llm = _llm_for_run()
    calls = []
    monkeypatch.setattr(
        llm_module,
        "run_tool_calling_llm",
        lambda llm_, params: calls.append(params) or iter(()),
    )
    monkeypatch.setattr(llm_module.litellm, "supports_vision", lambda model: False)

    def fail(model):
        raise providers.OpenRouterCatalogError("catalog down")

    monkeypatch.setattr(llm_module, "openrouter_model_entry", fail)
    messages = [
        {"role": "system", "type": "message", "content": "system"},
        {"role": "user", "type": "message", "content": "hello"},
    ]

    with pytest.raises(providers.OpenRouterCatalogError):
        list(llm.run(messages))

    assert calls == []
    assert llm.supports_vision is None


def test_run_caches_a_resolved_capability(monkeypatch):
    """A second run reuses the resolved value instead of fetching again."""
    llm = _llm_for_run()
    calls = {"count": 0}

    def detect(model):
        calls["count"] += 1
        return True

    monkeypatch.setattr(llm_module, "_supports_vision", detect)
    monkeypatch.setattr(
        llm_module,
        "run_tool_calling_llm",
        lambda llm_, params: iter(()),
    )
    messages = [
        {"role": "system", "type": "message", "content": "system"},
        {"role": "user", "type": "message", "content": "hello"},
    ]

    list(llm.run(messages))
    list(llm.run(messages))

    assert calls["count"] == 1


def test_model_change_resets_auto_detected_vision(monkeypatch):
    """A cached auto-detection must not apply to a different model."""
    llm = _llm_for_run()
    monkeypatch.setattr(llm_module, "_supports_vision", lambda model: True)
    monkeypatch.setattr(
        llm_module, "run_tool_calling_llm", lambda llm_, params: iter(())
    )
    messages = [
        {"role": "system", "type": "message", "content": "system"},
        {"role": "user", "type": "message", "content": "hello"},
    ]

    list(llm.run(messages))
    assert llm.supports_vision is True

    llm.model = "openrouter/vendor/other"

    assert llm.supports_vision is None


def test_model_change_preserves_explicit_vision():
    """An explicit supports_vision override survives a model change."""
    llm = _llm_for_run()
    llm.supports_vision = False

    llm.model = "openrouter/vendor/other"

    assert llm.supports_vision is False


def test_setting_the_same_model_keeps_cached_vision(monkeypatch):
    """Re-setting the same model must not discard a valid cached value."""
    llm = _llm_for_run()
    monkeypatch.setattr(llm_module, "_supports_vision", lambda model: True)
    monkeypatch.setattr(
        llm_module, "run_tool_calling_llm", lambda llm_, params: iter(())
    )
    messages = [
        {"role": "system", "type": "message", "content": "system"},
        {"role": "user", "type": "message", "content": "hello"},
    ]
    list(llm.run(messages))

    llm.model = llm.model

    assert llm.supports_vision is True
