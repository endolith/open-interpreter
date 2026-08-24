from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.terminal_interface.validate_llm_settings import (
    display_welcome_message_once,
    validate_llm_settings,
)
from interpreter.terminal_interface import validate_llm_settings as vls_module
from tests.helpers import TEST_LLM_MODEL


def test_validate_llm_settings_offline_breaks_immediately():
    """Offline mode skips API key prompts and returns without displaying messages."""
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
    """When OPENAI_API_KEY is set, validate_llm_settings does not prompt for a key."""
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
    """Welcome message is shown at most once per process."""
    if hasattr(display_welcome_message_once, "_displayed"):
        delattr(display_welcome_message_once, "_displayed")
    interpreter = SimpleNamespace(display_message=mock.Mock())
    with mock.patch("interpreter.terminal_interface.validate_llm_settings.time.sleep"):
        display_welcome_message_once(interpreter)
        display_welcome_message_once(interpreter)
    assert interpreter.display_message.call_count == 1
    if hasattr(display_welcome_message_once, "_displayed"):
        delattr(display_welcome_message_once, "_displayed")


def _interpreter(model="gpt-4o-mini", messages=None, api_key=None, api_base=None, **overrides):
    """Build a minimal interpreter stub with sane defaults for validate_llm_settings."""
    kwargs = {
        "offline": False,
        "auto_run": True,
        "messages": [] if messages is None else messages,
        "llm": SimpleNamespace(
            model=model,
            api_key=api_key,
            api_base=api_base,
            load=mock.Mock(),
        ),
        "display_message": mock.Mock(),
    }
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def test_validate_llm_settings_prompts_for_missing_key():
    """When no OpenAI key is available, validate_llm_settings prompts and stores the response."""
    interpreter = _interpreter()
    with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(
        vls_module, "display_welcome_message_once"
    ) as welcome, mock.patch.object(vls_module, "prompt", return_value="sk-abc") as prompt, mock.patch.object(
        vls_module.time, "sleep"
    ):
        validate_llm_settings(interpreter)
    welcome.assert_called_once_with(interpreter)
    prompt.assert_called_once_with("OpenAI API key: ", is_password=True)
    assert interpreter.llm.api_key == "sk-abc"


def test_validate_llm_settings_exits_on_local_command():
    """validate_llm_settings exits when the user types `interpreter --local` at the key prompt."""
    interpreter = _interpreter()
    with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(
        vls_module, "display_welcome_message_once"
    ), mock.patch.object(vls_module, "prompt", return_value="interpreter --local"), mock.patch.object(
        vls_module.time, "sleep"
    ), mock.patch("builtins.exit", side_effect=SystemExit) as exit_mock, mock.patch(
        "interpreter.terminal_interface.validate_llm_settings.print"
    ):
        with pytest.raises(SystemExit):
            validate_llm_settings(interpreter)
    exit_mock.assert_called_once_with()


def test_validate_llm_settings_uses_api_base_key():
    """validate_llm_settings does not prompt when an api_base is set (e.g. self-hosted proxy)."""
    interpreter = _interpreter(api_base="https://proxy.example.com")
    with mock.patch.dict("os.environ", {}, clear=True):
        validate_llm_settings(interpreter)
    assert interpreter.llm.api_key is None


def test_validate_llm_settings_uses_llm_api_key():
    """validate_llm_settings does not prompt when interpreter.llm.api_key is already set."""
    interpreter = _interpreter(api_key="sk-preset")
    with mock.patch.dict("os.environ", {}, clear=True):
        validate_llm_settings(interpreter)
    assert interpreter.llm.api_key == "sk-preset"


def test_validate_llm_settings_displays_model_set():
    """validate_llm_settings announces the active model for non-auto-run interactive use."""
    interpreter = _interpreter(
        model="custom-model", auto_run=False, messages=[{"role": "user"}, {"role": "assistant"}]
    )
    with mock.patch.dict("os.environ", {}, clear=True):
        validate_llm_settings(interpreter)
    interpreter.display_message.assert_called_with("> Model set to `custom-model`")


def test_validate_llm_settings_i_model_note():
    """validate_llm_settings shows the training-usage note when the model is `i`."""
    interpreter = _interpreter(model="i")
    with mock.patch.dict("os.environ", {}, clear=True):
        validate_llm_settings(interpreter)
    assert "will be used to train" in interpreter.display_message.call_args.args[0]


def test_validate_llm_settings_ollama_loads():
    """validate_llm_settings calls llm.load() for ollama model names."""
    interpreter = _interpreter(model="ollama/llama3", messages=[{"role": "user"}])
    with mock.patch.dict("os.environ", {}, clear=True):
        validate_llm_settings(interpreter)
    interpreter.llm.load.assert_called_once()
