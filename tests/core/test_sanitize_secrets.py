from interpreter.core.llm.utils.sanitize_secrets import _redact_secrets


def test_redacts_bare_fine_grained_github_token():
    """Bare fine-grained GitHub PATs are redacted without a secret-like variable name."""
    token = f"github_pat_{'A' * 22}_{'b' * 59}"
    text = f"observed {token} in command output"

    redacted = _redact_secrets(text)

    assert token not in redacted
    assert redacted.endswith("observed [REDACTED] in command output")


def test_does_not_redact_invalid_fine_grained_github_token_shape():
    """A malformed fine-grained GitHub token is preserved instead of over-redacted."""
    token = f"github_pat_{'A' * 21}_{'b' * 59}"
    text = f"observed {token} in command output"

    assert _redact_secrets(text) == text
