"""Per-route wire quirks, resolved once from the route's truthful identity.

Each entry is a deviation from the conventional OpenAI chat contract, and the
request builder is the only consumer. Adding a provider should mean adding a
row, not another `if "name" in model` branch: that inference is what made the
same class of bug reappear each time a model arrived under a new spelling.

Only quirks with an observable effect here belong in this table. Context/output
limits and tool/vision support come from LiteLLM and models.dev; this is for
request fields those catalogs cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quirks:
    """Wire deviations for one route, relative to OpenAI chat defaults."""

    # Some providers reject the `stream_options` object outright (Zhipu/GLM,
    # Cerebras). It only asks for usage accounting, so dropping it loses a
    # statistic, not a response.
    stream_options: bool = True

    # DashScope takes thinking controls in `extra_body`, not as OpenAI params:
    # `enable_thinking` (bool) and `thinking_budget` (int tokens).
    thinking_param: str | None = None
    thinking_budget_param: str | None = None

    # DashScope models with preserved thinking want the prior turns' reasoning
    # echoed plus this flag, or the model restarts its chain each turn.
    preserve_thinking_param: str | None = None


DEFAULT = Quirks()

_DASHSCOPE = Quirks(
    thinking_param="enable_thinking",
    thinking_budget_param="thinking_budget",
    preserve_thinking_param="preserve_thinking",
)

# Routes OI owns a prefix for.
_BY_ROUTE = {
    "dashscope-intl": _DASHSCOPE,
    "dashscope-us": _DASHSCOPE,
    "zhipu": Quirks(stream_options=False),
    "cerebras": Quirks(stream_options=False),
}

# Model families reached through a gateway whose route name says nothing about
# the upstream (opencode_go/glm-5.1 is OpenAI-shaped on the wire but Zhipu in
# its parameter rules). Match the family, never a full model id.
_BY_MODEL_FAMILY = {
    "glm": _BY_ROUTE["zhipu"],
}


def _family(model: str | None) -> str:
    """Leading token of the bare model id: `glm-5.1` -> `glm`, `qwen3-max` -> `qwen3`."""
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return bare.split("-", 1)[0].split(".", 1)[0]


def resolve_quirks(model: str | None, wire_model: str | None = None) -> Quirks:
    """Quirks for a route, falling back to OpenAI chat defaults.

    `model` is OI's truthful route name (``dashscope-intl/qwen3-max``); the
    fallback checks the wire id for gateways that hide the upstream in it.
    """
    route = (model or "").split("/", 1)[0].strip().lower()
    if route in _BY_ROUTE:
        return _BY_ROUTE[route]
    for candidate in (wire_model, model):
        quirks = _BY_MODEL_FAMILY.get(_family(candidate))
        if quirks is not None:
            return quirks
    return DEFAULT
