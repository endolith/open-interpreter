#!/usr/bin/env python3
"""One-shot non-streaming chat.completions to inspect raw usage (e.g. cached_tokens).

Env: DASHSCOPE_API_KEY

Example (Git Bash):  export DASHSCOPE_API_KEY='sk-...' && python scripts/dashscope_usage_probe.py
Example (cmd):       set DASHSCOPE_API_KEY=sk-... && python scripts/dashscope_usage_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"


def main() -> None:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        print("Set DASHSCOPE_API_KEY first.", file=sys.stderr)
        sys.exit(1)

    payload = json.dumps(
        {
            "model": "qwen3.5-plus",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        URL,
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

    print(f"HTTP {status}")
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
