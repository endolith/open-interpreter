"""Read token usage from LiteLLM streaming chunks (e.g. OpenAI stream_options include_usage)."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_plain(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _to_plain(value.dict())
        except Exception:
            pass
    model_dump_json = getattr(value, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            return json.loads(model_dump_json())
        except Exception:
            pass
    return str(value)


def _usage_from_chunk(chunk: Any) -> Optional[dict]:
    if chunk is None:
        return None
    usage = None
    if isinstance(chunk, Mapping):
        usage = chunk.get("usage")
    else:
        usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    plain = _to_plain(usage)
    if isinstance(plain, dict) and plain:
        return plain
    return None


def record_stream_chunk_usage(llm, chunk: Any) -> None:
    """
    If this stream chunk carries a usage object, store it on llm.last_completion_usage.
    OpenAI-compatible APIs often attach usage only to the final streaming chunk.
    """
    usage = _usage_from_chunk(chunk)
    if usage:
        llm.last_completion_usage = usage


def format_last_usage_markdown(usage: dict) -> str:
    header = (
        "> **Last LLM completion** (most recent API call; each tool round is a separate call):\n\n"
    )
    body: list[str] = []
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")
    if pt is not None:
        body.append(f"- **Prompt tokens:** {pt}\n")
    if ct is not None:
        body.append(f"- **Completion tokens:** {ct}\n")
    if tt is not None:
        body.append(f"- **Total tokens:** {tt}\n")

    details = usage.get("prompt_tokens_details")
    if details:
        d = _to_plain(details) if not isinstance(details, dict) else details
        if isinstance(d, dict):
            for k, v in sorted(d.items()):
                label = k.replace("_", " ").title()
                body.append(f"- **Prompt — {label}:** {v}\n")

    shown = {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details"}
    rest = {k: v for k, v in usage.items() if k not in shown and v not in (None, {})}
    if rest:
        body.append("\n**Other usage fields:**\n\n```json\n")
        body.append(json.dumps(_to_plain(rest), indent=2))
        body.append("\n```\n")

    if not body:
        body.append(f"```json\n{json.dumps(_to_plain(usage), indent=2)}\n```\n")

    return header + "".join(body)
