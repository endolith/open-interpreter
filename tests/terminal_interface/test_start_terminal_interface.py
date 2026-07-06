import os
from contextlib import ExitStack, contextmanager
from unittest import mock

from interpreter import OpenInterpreter
from interpreter.terminal_interface.start_terminal_interface import (
    apply_telemetry_env_overrides,
    get_argument_dictionary,
    set_attributes,
    start_terminal_interface,
)


def _disable_telemetry_argument():
    """Return the CLI argument definition for disable_telemetry."""
    interpreter = OpenInterpreter()
    arguments = [
        {
            "name": "disable_telemetry",
            "type": bool,
            "action": "BooleanOptionalAction",
            "default": True,
            "attribute": {"object": interpreter, "attr_name": "disable_telemetry"},
        }
    ]
    return interpreter, arguments


@contextmanager
def _offline_cli_context(interpreter, argv, env=None):
    """Run start_terminal_interface with --offline and post-setup hooks mocked out."""
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch(
                "interpreter.terminal_interface.start_terminal_interface.validate_llm_settings"
            )
        )
        stack.enter_context(
            mock.patch(
                "interpreter.terminal_interface.start_terminal_interface.check_for_update",
                return_value=False,
            )
        )
        stack.enter_context(
            mock.patch(
                "interpreter.terminal_interface.start_terminal_interface.profile",
                side_effect=lambda inst, profile_name: inst,
            )
        )
        stack.enter_context(
            mock.patch(
                "interpreter.terminal_interface.start_terminal_interface.contribute_conversation_launch_logic"
            )
        )
        stack.enter_context(mock.patch.object(interpreter, "chat"))
        stack.enter_context(mock.patch("sys.argv", argv))
        stack.enter_context(mock.patch.dict(os.environ, env or {}, clear=False))
        yield


def test_apply_telemetry_env_overrides_respects_disable_telemetry_true(monkeypatch):
    """DISABLE_TELEMETRY=true forces telemetry off even when the CLI opted in."""
    interpreter = OpenInterpreter(disable_telemetry=False)
    monkeypatch.setenv("DISABLE_TELEMETRY", "true")

    apply_telemetry_env_overrides(interpreter)

    assert interpreter.disable_telemetry is True


def test_apply_telemetry_env_overrides_disable_telemetry_false_enables(monkeypatch):
    """DISABLE_TELEMETRY=false opts in to telemetry when that env var is set."""
    interpreter = OpenInterpreter()
    monkeypatch.setenv("DISABLE_TELEMETRY", "false")

    apply_telemetry_env_overrides(interpreter)

    assert interpreter.disable_telemetry is False


def test_apply_telemetry_env_overrides_enable_telemetry_env_var(monkeypatch):
    """ENABLE_TELEMETRY=true opts in when DISABLE_TELEMETRY is not set."""
    interpreter = OpenInterpreter()
    monkeypatch.delenv("DISABLE_TELEMETRY", raising=False)
    monkeypatch.setenv("ENABLE_TELEMETRY", "true")

    apply_telemetry_env_overrides(interpreter)

    assert interpreter.disable_telemetry is False


def test_apply_telemetry_env_overrides_no_env_leaves_interpreter_unchanged(monkeypatch):
    """Without telemetry env vars, the interpreter keeps its existing setting."""
    interpreter = OpenInterpreter(disable_telemetry=False)
    monkeypatch.delenv("DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("ENABLE_TELEMETRY", raising=False)

    apply_telemetry_env_overrides(interpreter)

    assert interpreter.disable_telemetry is False


def test_cli_telemetry_disabled_by_default():
    """The CLI leaves telemetry off when no flags or env vars override it."""
    interpreter = OpenInterpreter()
    with _offline_cli_context(interpreter, ["interpreter", "--offline"]):
        start_terminal_interface(interpreter)
    assert interpreter.disable_telemetry is True


def test_cli_no_disable_telemetry_flag_enables_telemetry():
    """--no-disable_telemetry opts in to anonymous telemetry from the CLI."""
    interpreter = OpenInterpreter()
    with _offline_cli_context(interpreter, ["interpreter", "--offline", "--no-disable_telemetry"]):
        start_terminal_interface(interpreter)
    assert interpreter.disable_telemetry is False


def test_cli_disable_telemetry_env_var_overrides_cli_flag(monkeypatch):
    """DISABLE_TELEMETRY env var wins over --no-disable_telemetry."""
    interpreter = OpenInterpreter()
    monkeypatch.setenv("DISABLE_TELEMETRY", "true")
    with _offline_cli_context(
        interpreter,
        ["interpreter", "--offline", "--no-disable_telemetry"],
    ):
        start_terminal_interface(interpreter)
    assert interpreter.disable_telemetry is True


def test_set_attributes_applies_disable_telemetry_default():
    """set_attributes applies the CLI default of disable_telemetry=True."""
    interpreter, arguments = _disable_telemetry_argument()
    args = mock.Mock(disable_telemetry=True, verbose=False)

    set_attributes(args, arguments)

    assert interpreter.disable_telemetry is True


def test_get_argument_dictionary_returns_matching_entry():
    """get_argument_dictionary returns the argument metadata for a known flag."""
    arguments = [{"name": "disable_telemetry", "default": True}]

    result = get_argument_dictionary(arguments, "disable_telemetry")

    assert result == arguments[0]


def test_get_argument_dictionary_returns_empty_for_unknown_key():
    """get_argument_dictionary returns an empty dict for unknown CLI arguments."""
    assert get_argument_dictionary([], "missing") == {}
