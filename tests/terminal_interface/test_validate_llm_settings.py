from types import SimpleNamespace
from unittest import mock

from interpreter.terminal_interface.validate_llm_settings import (
    display_welcome_message_once,
    validate_llm_settings,
)

from tests.conftest import TEST_LLM_MODEL


def test_validate_llm_settings_offline_breaks_immediately():
    interpreter = SimpleNamespace(
        offline=True,
        auto_run=False,
        messages=[],
        # Model name unused on offline path; only llm.load may be referenced.
        llm=SimpleNamespace(model=TEST_LLM_MODEL, load=mock.Mock()),
        display_message=mock.Mock(),
    )
    validate_llm_settings(interpreter)
    interpreter.display_message.assert_not_called()


def test_validate_llm_settings_with_env_key_skips_prompt():
    # TEST_LLM_MODEL is in validate_llm_settings' OpenAI model list, so missing
    # keys would normally prompt — unless OPENAI_API_KEY is already set.
    interpreter = SimpleNamespace(
        offline=False,
        auto_run=True,
        messages=[],
        llm=SimpleNamespace(
            model=TEST_LLM_MODEL,
            api_key=None,
            api_base=None,
            load=mock.Mock(),
        ),
        display_message=mock.Mock(),
    )
    with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        validate_llm_settings(interpreter)
    interpreter.display_message.assert_not_called()


def test_display_welcome_message_once_only_first_time():
    if hasattr(display_welcome_message_once, "_displayed"):
        delattr(display_welcome_message_once, "_displayed")
    interpreter = SimpleNamespace(display_message=mock.Mock())
    with mock.patch("interpreter.terminal_interface.validate_llm_settings.time.sleep"):
        display_welcome_message_once(interpreter)
        display_welcome_message_once(interpreter)
    assert interpreter.display_message.call_count == 1
    if hasattr(display_welcome_message_once, "_displayed"):
        delattr(display_welcome_message_once, "_displayed")
