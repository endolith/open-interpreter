"""
Custom detect_secrets plugin: redact values of env-style NAME=VALUE lines when
NAME ends with a secret-like suffix (_KEY, _SECRET, _TOKEN, _PASSWORD, etc.).
Used so os.environ / .env dumps are sanitized even when format-specific
detectors miss.
"""
import re
from typing import Generator

from detect_secrets.plugins.base import BasePlugin


# Env var name suffixes that indicate the value is a secret (convention-based).
ENV_SECRET_SUFFIXES = (
    "_KEY",
    "_SECRET",
    "_TOKEN",
    "_PASSWORD",
    "_PASS",
    "_PWD",
    "_CLIENT_ID",  # OAuth client IDs (e.g. OSM_OAUTH_CLIENT_ID) can be sensitive
)

# Shell/env: NAME=VALUE or NAME = VALUE
_ENV_LINE_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)\s*$")
# Python repr(os.environ): 'NAME': 'value' or "NAME": "value" (may appear multiple times per line)
_PYTHON_ENV_RE = re.compile(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*:\s*['\"]([^\'\"]*)['\"]")


class EnvSecretDetector(BasePlugin):
    """
    Detects secret values in env-style lines (NAME=VALUE) when NAME ends with a
    secret-like suffix. Yields only the value so the library redacts it, not
    the line.
    """

    secret_type = "Env Secret"

    def analyze_string(self, string: str, **kwargs: object) -> Generator[str, None, None]:
        def maybe_yield(name: str, value: str) -> Generator[str, None, None]:
            if not any(name.upper().endswith(s) for s in ENV_SECRET_SUFFIXES):
                return
            if value:
                yield value

        m = _ENV_LINE_RE.match(string)
        if m:
            name, value = m.group(1), m.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            yield from maybe_yield(name, value)
            return

        for m in _PYTHON_ENV_RE.finditer(string):
            name, value = m.group(1), m.group(2)
            yield from maybe_yield(name, value)
