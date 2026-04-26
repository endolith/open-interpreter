#!/usr/bin/env python3
"""One-shot non-streaming chat.completions to inspect raw usage (e.g. cached_tokens).

Env: DASHSCOPE_API_KEY

Optional arg: model id (default dashscope-us/qwen3.5-plus). On compatible-mode we have seen
qwen3-max return prompt_tokens_details.cached_tokens (e.g. 0) while qwen3.5-plus sometimes
omits cached_tokens entirely in the same usage object — provider/model-dependent.

Use model slug prefixes to choose region endpoint, matching interpreter behavior:
- `dashscope-us/<model>` (US Virginia)
- `dashscope-intl/<model>` (Singapore International)

Examples (Git Bash):
  export DASHSCOPE_API_KEY='sk-...' && python scripts/dashscope_usage_probe.py dashscope-us/qwen3-max
  export DASHSCOPE_API_KEY='sk-...' && python scripts/dashscope_usage_probe.py dashscope-intl/qwen3-max
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

# US (Virginia) (us-east-1)
URL_US = "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"
# Singapore (ap-southeast-1)
URL_INTL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"


def main() -> None:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        print("Set DASHSCOPE_API_KEY first.", file=sys.stderr)
        sys.exit(1)

    model_slug = sys.argv[1] if len(sys.argv) > 1 else "dashscope-us/qwen3.5-plus"
    model_slug_lower = model_slug.lower()
    if model_slug_lower.startswith("dashscope-intl/"):
        model = model_slug.split("/", 1)[1]
        url = URL_INTL
        region = "intl"
    elif model_slug_lower.startswith("dashscope-us/"):
        model = model_slug.split("/", 1)[1]
        url = URL_US
        region = "us"
    else:
        print(
            "Model must start with dashscope-us/ or dashscope-intl/ (e.g. dashscope-us/qwen3-max).",
            file=sys.stderr,
        )
        sys.exit(2)

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}", file=sys.stderr)
        err_body = e.read().decode("utf-8", errors="replace")
        print(err_body[:4000])
        sys.exit(1)

    print(f"HTTP {status}  model={model_slug}  region={region}  url={url}")
    data = json.loads(body)
    if "error" in data:
        print(json.dumps(data, indent=2))
        sys.exit(1)

    usage = data.get("usage")
    print("--- usage (raw) ---")
    print(json.dumps(usage, indent=2))
    print("--- top-level keys ---")
    print(sorted(data.keys()))


if __name__ == "__main__":
    main()
