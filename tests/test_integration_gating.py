import os
from unittest import mock

from conftest import (
    _INTEGRATION_API_KEY_SKIP,
    _INTEGRATION_OPT_IN_SKIP,
    integration_skip_reason,
)


def test_integration_skip_without_opt_in():
    """Integration tests skip without OI_RUN_INTEGRATION even when OPENAI_API_KEY is set."""
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
        assert integration_skip_reason() == _INTEGRATION_OPT_IN_SKIP


def test_integration_skip_without_api_key():
    """Integration tests skip when OPENAI_API_KEY is unset even with OI_RUN_INTEGRATION=1."""
    with mock.patch.dict(os.environ, {"OI_RUN_INTEGRATION": "1"}, clear=True):
        assert integration_skip_reason() == _INTEGRATION_API_KEY_SKIP


def test_integration_allowed_with_opt_in_and_api_key():
    """Integration tests run only when both OI_RUN_INTEGRATION=1 and OPENAI_API_KEY are set."""
    with mock.patch.dict(
        os.environ,
        {"OI_RUN_INTEGRATION": "1", "OPENAI_API_KEY": "sk-test"},
        clear=True,
    ):
        assert integration_skip_reason() is None


def test_groq_model_requires_groq_key():
    """A groq model checks GROQ_API_KEY, not the OpenAI fallback."""
    with mock.patch.dict(
        os.environ,
        {
            "OI_RUN_INTEGRATION": "1",
            "OI_INTEGRATION_MODEL": "groq/llama-3.1-8b-instant",
            "OPENAI_API_KEY": "sk-test",
        },
        clear=True,
    ):
        reason = integration_skip_reason()

    assert reason is not None
    assert "GROQ_API_KEY" in reason


def test_groq_model_allowed_with_groq_key():
    """The provider key for the selected model is enough to run."""
    with mock.patch.dict(
        os.environ,
        {
            "OI_RUN_INTEGRATION": "1",
            "OI_INTEGRATION_MODEL": "groq/llama-3.1-8b-instant",
            "GROQ_API_KEY": "gsk-test",
        },
        clear=True,
    ):
        assert integration_skip_reason() is None


def test_local_model_needs_no_key():
    """Ollama models need no provider key, so the gate allows them."""
    with mock.patch.dict(
        os.environ,
        {
            "OI_RUN_INTEGRATION": "1",
            "OI_INTEGRATION_MODEL": "ollama/llama3",
        },
        clear=True,
    ):
        assert integration_skip_reason() is None
