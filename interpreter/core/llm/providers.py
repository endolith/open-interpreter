"""Provider metadata that litellm's registry does not know yet."""

import os

import requests

# Parsed OpenRouter /api/v1/models list, fetched at most once per process.
# None means "not fetched yet"; a failed fetch leaves it None so a transient
# error does not disable the model for the rest of the process.
_openrouter_models = None


class OpenRouterCatalogError(RuntimeError):
    """The OpenRouter model catalog could not be fetched or parsed."""


def openrouter_model_entry(model: str):
    """Return the OpenRouter catalog entry for an ``openrouter/`` model.

    OpenRouter proxies models that litellm has not registered, and its catalog
    is authoritative for input modalities. The whole list is fetched once and
    cached; the lookup matches the exact slug after ``openrouter/`` so one
    route's capabilities are never inferred from another route's id.

    Returns None when the model is not an OpenRouter one or the slug is absent.
    Raises OpenRouterCatalogError when the catalog cannot be fetched or parsed,
    so the caller can abort the request instead of treating the model as
    text-only.
    """
    global _openrouter_models
    prefix = "openrouter/"
    if not model.lower().startswith(prefix):
        return None
    if _openrouter_models is None:
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={
                    "HTTP-Referer": os.environ.get("OR_SITE_URL", ""),
                    "X-Title": os.environ.get("OR_APP_NAME", "Open Interpreter"),
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            raise OpenRouterCatalogError(
                "could not fetch the OpenRouter model catalog"
            ) from error
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise OpenRouterCatalogError(
                "OpenRouter model catalog response has no model list"
            )
        _openrouter_models = {
            entry["id"]: entry
            for entry in data
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        }
    return _openrouter_models.get(model[len(prefix):])
