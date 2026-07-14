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
