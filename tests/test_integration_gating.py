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
