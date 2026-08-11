import sys
from unittest import mock

from interpreter.terminal_interface.start_terminal_interface import (
    get_argument_dictionary,
    set_attributes,
    start_terminal_interface,
)


def test_get_argument_dictionary_returns_matching_entry():
    """get_argument_dictionary resolves an argument spec by its name."""
    arguments = [
        {"name": "model", "nickname": "m", "type": str},
        {"name": "verbose", "nickname": "v", "type": bool},
    ]
    assert get_argument_dictionary(arguments, "model")["nickname"] == "m"


def test_get_argument_dictionary_unknown_returns_empty():
    """get_argument_dictionary returns an empty dict for an unknown argument name."""
    assert get_argument_dictionary([{"name": "model"}], "nope") == {}


class _Obj:
    """Plain namespace so hasattr() reflects only what set_attributes actually wrote."""


def _obj():
    obj = _Obj()
    obj.llm = _Obj()
    return obj


def test_set_attributes_applies_non_none_values():
    """set_attributes copies CLI argument values onto the objects named by each spec."""
    interpreter = _obj()
    arguments = [
        {
            "name": "verbose",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "verbose"},
        },
        {
            "name": "model",
            "type": str,
            "attribute": {"object": interpreter.llm, "attr_name": "model"},
        },
        {"name": "safe_mode", "type": str},  # No "attribute" key -> must be ignored
    ]

    class Args:
        pass

    args = Args()
    args.verbose = True
    args.model = "gpt-4"
    args.safe_mode = "off"

    set_attributes(args, arguments)

    assert interpreter.verbose is True
    assert interpreter.llm.model == "gpt-4"
    # Arguments without an "attribute" mapping are never applied.
    assert not hasattr(interpreter, "safe_mode")


def test_set_attributes_skips_none_values():
    """set_attributes does not apply arguments whose value is None."""
    interpreter = _obj()
    arguments = [
        {
            "name": "verbose",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "verbose"},
        }
    ]

    class Args:
        pass

    args = Args()
    args.verbose = None

    set_attributes(args, arguments)

    # Nothing was written onto the interpreter.
    assert not hasattr(interpreter, "verbose")


def test_start_terminal_interface_version_flag_returns_early(monkeypatch, capsys):
    """`--version` prints the version and returns before starting a chat session."""
    monkeypatch.setattr(sys, "argv", ["oi", "--version"])
    interpreter = mock.MagicMock()

    result = start_terminal_interface(interpreter)

    assert result is None
    assert "Open Interpreter" in capsys.readouterr().out


def test_start_terminal_interface_renames_deprecated_debug_mode_flag(
    monkeypatch, capsys
):
    """The deprecated `--debug_mode` flag is rewritten to `--verbose` and parsing continues."""
    monkeypatch.setattr(sys, "argv", ["oi", "--debug_mode", "--version"])
    interpreter = mock.MagicMock()

    result = start_terminal_interface(interpreter)

    output = capsys.readouterr().out
    assert "`--debug_mode` has been renamed to `--verbose`" in output
    assert result is None


def _patch_module(monkeypatch):
    """Patch the CLI helper imports so a call can't touch profiles, network, or
    the LLM."""
    import interpreter.terminal_interface.start_terminal_interface as sti

    interpreter = mock.Mock()
    interpreter.auto_run = False
    interpreter.safe_mode = "off"
    interpreter.offline = True  # skips the update check
    interpreter.messages = []
    interpreter.disable_telemetry = False
    interpreter.chat = mock.Mock()
    # Concrete values so the model/api_base string logic sees real strings.
    interpreter.llm.model = "gpt-4o-mini"
    interpreter.llm.api_base = None
    interpreter.llm.context_window = None
    interpreter.llm.max_tokens = None
    interpreter.llm.supports_functions = None

    monkeypatch.setattr(sti, "profile", mock.Mock(side_effect=lambda i, p: i))
    monkeypatch.setattr(sti, "open_storage_dir", mock.Mock())
    monkeypatch.setattr(sti, "reset_profile", mock.Mock())
    monkeypatch.setattr(sti, "check_for_update", mock.Mock(return_value=False))
    monkeypatch.setattr(sti, "validate_llm_settings", mock.Mock())
    monkeypatch.setattr(sti, "conversation_navigator", mock.Mock())
    monkeypatch.setattr(
        sti, "contribute_conversation_launch_logic", mock.Mock()
    )
    monkeypatch.setattr(sti.time, "sleep", mock.Mock())
    monkeypatch.setattr(sti, "version", mock.Mock(return_value="0.2.5"))
    return sti, interpreter


def test_start_terminal_interface_profiles_flag_opens_storage(monkeypatch):
    """`--profiles` opens the profiles storage directory and returns early."""
    sti, interpreter = _patch_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["oi", "--profiles"])

    assert start_terminal_interface(interpreter) is None
    sti.open_storage_dir.assert_called_once_with("profiles")
    interpreter.chat.assert_not_called()


def test_start_terminal_interface_local_models_flag_opens_storage(monkeypatch):
    """`--local_models` opens the models storage directory and returns early."""
    sti, interpreter = _patch_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["oi", "--local_models"])

    assert start_terminal_interface(interpreter) is None
    sti.open_storage_dir.assert_called_once_with("models")


def test_start_terminal_interface_reset_profile_flag(monkeypatch):
    """`--reset_profile <name>` resets that profile and returns early."""
    sti, interpreter = _patch_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["oi", "--reset_profile", "default.yaml"])

    assert start_terminal_interface(interpreter) is None
    sti.reset_profile.assert_called_once_with("default.yaml")


def test_start_terminal_interface_fast_shortcut_selects_fast_profile(monkeypatch):
    """`--fast` loads the fast.yaml profile instead of the default."""
    sti, interpreter = _patch_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["oi", "--fast"])

    start_terminal_interface(interpreter)

    assert sti.profile.call_args[0][1] == "fast.yaml"


def test_start_terminal_interface_gpt4_sets_sensible_defaults(monkeypatch):
    """A bare `gpt-4` model gets a 6500-token context window and 4096 max tokens."""
    sti, interpreter = _patch_module(monkeypatch)
    interpreter.llm.model = "gpt-4"
    interpreter.llm.context_window = None
    interpreter.llm.max_tokens = None
    interpreter.llm.supports_functions = None
    monkeypatch.setattr(sys, "argv", ["oi"])

    start_terminal_interface(interpreter)

    assert interpreter.llm.context_window == 6500
    assert interpreter.llm.max_tokens == 4096
    assert interpreter.llm.supports_functions is True


def test_start_terminal_interface_api_base_prefixes_openai_model(monkeypatch):
    """A custom api_base rewrites a bare model name to the openai/ namespace."""
    sti, interpreter = _patch_module(monkeypatch)
    interpreter.llm.api_base = "http://localhost:8000/v1"
    interpreter.llm.model = "gpt-4o-mini"
    monkeypatch.setattr(sys, "argv", ["oi"])

    start_terminal_interface(interpreter)

    assert interpreter.llm.model == "openai/gpt-4o-mini"


def test_start_terminal_interface_api_base_strips_jan_prefix(monkeypatch):
    """A jan/ model name has its prefix stripped when an api_base is set."""
    sti, interpreter = _patch_module(monkeypatch)
    interpreter.llm.api_base = "http://localhost:1234/v1"
    interpreter.llm.model = "jan/qwen2.5"
    monkeypatch.setattr(sys, "argv", ["oi"])

    start_terminal_interface(interpreter)

    assert interpreter.llm.model == "qwen2.5"


def test_start_terminal_interface_remaps_claude_35_model(monkeypatch):
    """Legacy claude-3.5 model names are remapped to claude-sonnet-4-6."""
    sti, interpreter = _patch_module(monkeypatch)
    interpreter.llm.model = "claude-3.5"
    interpreter.llm.api_base = None
    monkeypatch.setattr(sys, "argv", ["oi"])

    start_terminal_interface(interpreter)

    assert interpreter.llm.model == "claude-sonnet-4-6"


def test_start_terminal_interface_safe_mode_disables_auto_run(monkeypatch):
    """A safe mode of ask/auto turns off auto_run so code still needs approval."""
    sti, interpreter = _patch_module(monkeypatch)
    interpreter.auto_run = True
    interpreter.safe_mode = "ask"
    monkeypatch.setattr(sys, "argv", ["oi"])

    start_terminal_interface(interpreter)

    assert interpreter.auto_run is False


def test_start_terminal_interface_stdin_mode_chats_with_input(monkeypatch):
    """`--stdin` reads one line and passes it straight to interpreter.chat."""
    sti, interpreter = _patch_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["oi", "--stdin"])
    monkeypatch.setattr("builtins.input", lambda: "hello")

    start_terminal_interface(interpreter)

    assert interpreter.plain_text_display is True
    interpreter.chat.assert_called_once_with("hello")


def test_start_terminal_interface_i_shortcut_prepends_command(monkeypatch, tmp_path):
    """A bare first argument runs as an ultra-fast single-shot command."""
    sti, interpreter = _patch_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["oi", "make a pomodoro"])
    monkeypatch.chdir(tmp_path)

    start_terminal_interface(interpreter)

    assert interpreter.messages[0]["content"] == "I make a pomodoro"
    assert interpreter.custom_instructions.startswith("UPDATED INSTRUCTIONS")
    assert sys.argv == ["oi"]
