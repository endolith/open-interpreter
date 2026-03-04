"""
Sanitize message contents before sending to API LLMs: redact secrets (API keys,
passwords, etc.) that may appear in code output (e.g. from printing env vars).
Uses bc-detect-secrets (Bridgecrew). Not applied when using local models by default.

We use only a subset of detect_secrets plugins so that paths, UUIDs, and other
benign high-entropy strings in code output are not redacted (see PLUGINS_EXCLUDED_FOR_SANITIZE).
EnvSecretDetector (env_secret_detector.py) is called directly (not via library file-loading)
to reliably redact env-style NAME=VALUE lines whose names end in _KEY, _SECRET, _TOKEN, etc.
"""

# The library's full default set includes:
#
# DESIRED (we use these – explicit API keys / tokens / credentials, or keyword-based value only):
#   OpenAIDetector, AWSKeyDetector, AzureStorageKeyDetector, BasicAuthDetector,
#   ArtifactoryDetector, CloudantDetector, DiscordBotTokenDetector, GitHubTokenDetector,
#   GitLabTokenDetector (if present), IbmCloudIamDetector, IbmCosHmacDetector,
#   JwtTokenDetector, KeywordDetector, MailchimpDetector, NpmDetector, PypiTokenDetector (if present),
#   PrivateKeyDetector, SendGridDetector, SlackDetector, SoftlayerDetector,
#   SquareOAuthDetector, StripeDetector, TelegramBotTokenDetector (if present), TwilioKeyDetector.
#   (KeywordDetector redacts only the value after password=, api_key=, etc., not the whole line.)
#
# NOT DESIRED (we exclude these – too aggressive for env/output dumps):
#   Base64HighEntropyString – matches any high-entropy base64 (paths, GUIDs, benign strings).
#   HexHighEntropyString – matches UUIDs, hex segments in paths, etc.
#   IPPublicDetector – if present; public IPs in output are often not secrets.
#
PLUGINS_EXCLUDED_FOR_SANITIZE = frozenset({
    "Base64HighEntropyString",
    "HexHighEntropyString",
    "IPPublicDetector",
})


def _get_sanitize_plugins_config():
    """Return plugins_used config with credential/API-key detectors; excludes high-entropy and IP detectors.
    EnvSecretDetector is NOT included here – it is called directly in _redact_secrets to avoid
    the library's file-loading path (file:// URL resolution, lru_cache ordering, etc.)."""
    from detect_secrets.core.plugins.util import get_mapping_from_secret_type_to_class

    mapping = get_mapping_from_secret_type_to_class()
    return [
        {"name": cls.__name__}
        for cls in mapping.values()
        if cls.__name__ not in PLUGINS_EXCLUDED_FOR_SANITIZE
    ]


def _is_local_model(model: str) -> bool:
    """
    Return True if the model is typically run locally (ollama, local server, etc.).
    Used to decide whether to skip secrets sanitization by default.
    """
    if not model:
        return False
    m = model.lower()
    if m.startswith("ollama/"):
        return True
    if m.startswith("local/"):
        return True
    if m.startswith("jan/"):
        return True
    return False


def _redact_secrets(text: str) -> str:
    """
    Use bc-detect-secrets to find and redact secrets in text.
    Scans line-by-line and replaces each detected secret value with [REDACTED].
    Uses only credential/API-key detectors (not high-entropy or keyword) so paths
    and benign strings in code output are left visible.
    EnvSecretDetector is called directly (not via the library's file-loading mechanism)
    so it reliably runs regardless of path resolution or lru_cache ordering.
    """
    if not text or not isinstance(text, str):
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    from detect_secrets.core.scan import scan_line
    from detect_secrets.settings import transient_settings

    from .env_secret_detector import EnvSecretDetector
    _env_detector = EnvSecretDetector()

    config = {"plugins_used": _get_sanitize_plugins_config()}
    secrets_found = set()
    with transient_settings(config):
        from detect_secrets.core.plugins import util as _plugins_util
        _plugins_util.get_mapping_from_secret_type_to_class.cache_clear()
        for line in text.split("\n"):
            for secret in scan_line(line):
                if getattr(secret, "secret_value", None):
                    secrets_found.add(secret.secret_value)
            for value in _env_detector.analyze_string(line):
                secrets_found.add(value)

    if not secrets_found:
        return text

    result = text
    for s in sorted(secrets_found, key=len, reverse=True):
        result = result.replace(s, "[REDACTED]")
    disclaimer = "[Secret values in the output below were redacted before being sent to you; the original content is not literal.]\n\n"
    return disclaimer + result


def sanitize_messages(messages: list, scanner=None, only_code_output: bool = True) -> None:
    """
    Mutate message contents in place, redacting secrets in any text.
    Only text fields are scanned; images and other content are left unchanged.
    If only_code_output is True (default), only messages that are code execution
    results (role "function" with name "execute") are sanitized. This avoids
    redacting system prompt, user text, or assistant text, which can confuse
    the model when detectors match benign phrases.
    """
    redact = scanner if scanner else _redact_secrets
    for message in messages:
        if only_code_output:
            if message.get("role") != "function" or message.get("name") != "execute":
                continue
        content = message.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            message["content"] = redact(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                    part["text"] = redact(part["text"])


def should_sanitize_for_model(model: str, sanitize_secrets_setting) -> bool:
    """
    Return True if we should run secrets sanitization before sending to the LLM.
    sanitize_secrets_setting: "auto" | "on" | "off" | True | False (None treated as "auto").
    - "auto" / None: sanitize when model is not local.
    - "on" / True: always sanitize.
    - "off" / False: never sanitize.
    """
    if sanitize_secrets_setting in (True, "on"):
        return True
    if sanitize_secrets_setting in (False, "off"):
        return False
    return not _is_local_model(model)
