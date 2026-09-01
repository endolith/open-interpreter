from types import SimpleNamespace
from unittest import mock

import pytest
import sys

from interpreter.terminal_interface import magic_commands
from tests.helpers import TEST_LLM_MODEL


def _interpreter(**kwargs):
    """Build a SimpleNamespace interpreter stub, overriding defaults with kwargs."""
    base = {
        "messages": [],
        "system_message": "sys",
        "verbose": False,
        "debug": False,
        "auto_run": True,
        "llm": SimpleNamespace(model=TEST_LLM_MODEL),
        "display_message": mock.Mock(),
        "reset": mock.Mock(),
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_handle_undo_removes_messages_after_last_user():
    """%undo drops the last user turn and every message after it."""
    interpreter = _interpreter(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "reply2"},
        ]
    )
    magic_commands.handle_undo(interpreter, "")
    assert interpreter.messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply1"},
    ]


def test_handle_undo_noop_on_empty_messages():
    """%undo leaves an empty message list unchanged."""
    interpreter = _interpreter(messages=[])
    magic_commands.handle_undo(interpreter, "")
    assert interpreter.messages == []


def test_handle_reset_calls_reset():
    """%reset delegates to interpreter.reset()."""
    interpreter = _interpreter()
    magic_commands.handle_reset(interpreter, "")
    interpreter.reset.assert_called_once()


def test_handle_verbose_toggles_flag():
    """%verbose true/false toggles interpreter.verbose."""
    interpreter = _interpreter(verbose=False)
    magic_commands.handle_verbose(interpreter, "true")
    assert interpreter.verbose is True
    magic_commands.handle_verbose(interpreter, "false")
    assert interpreter.verbose is False


def test_handle_auto_run_toggles_flag():
    """%auto_run true enables interpreter.auto_run."""
    interpreter = _interpreter(auto_run=False)
    magic_commands.handle_auto_run(interpreter, "true")
    assert interpreter.auto_run is True


def test_handle_save_and_load_message_round_trip(tmp_path):
    """Saved messages can be reloaded from disk with %save_message and %load_message."""
    interpreter = _interpreter(messages=[{"role": "user", "content": "saved"}])
    path = tmp_path / "msgs.json"
    magic_commands.handle_save_message(interpreter, str(path))
    interpreter.messages = []
    magic_commands.handle_load_message(interpreter, str(path))
    assert interpreter.messages == [{"role": "user", "content": "saved"}]


def test_get_downloads_path_uses_home_on_posix(monkeypatch, tmp_path):
    """Non-Windows path uses expanduser('~')/Downloads and creates the folder."""
    monkeypatch.setattr(magic_commands.os, "name", "posix")
    monkeypatch.setattr(
        magic_commands.os.path,
        "expanduser",
        lambda path: str(tmp_path) if path == "~" else path,
    )
    downloads = magic_commands.get_downloads_path()
    assert downloads == str(tmp_path / "Downloads")
    assert (tmp_path / "Downloads").exists()


@pytest.mark.windows_ci
def test_get_downloads_path_windows(monkeypatch, tmp_path):
    """On Windows, get_downloads_path uses USERPROFILE/Downloads and creates the folder."""
    monkeypatch.setattr(magic_commands.os, "name", "nt")
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    downloads = magic_commands.get_downloads_path()
    assert downloads == str(tmp_path / "Downloads")


def test_handle_count_tokens_displays_estimate(capsys):
    """%tokens displays a token estimate for the current conversation."""
    interpreter = _interpreter(messages=[{"role": "user", "type": "message", "content": "hi"}])
    with mock.patch(
        "interpreter.terminal_interface.magic_commands.count_messages_tokens",
        return_value=(10, 0.001),
    ):
        magic_commands.handle_count_tokens(interpreter, "")
    interpreter.display_message.assert_called()
    assert "Tokens sent" in interpreter.display_message.call_args[0][0]


def test_handle_count_tokens_with_prompt():
    """%tokens with an argument includes that prompt in the next-request total."""
    interpreter = _interpreter(messages=[{"role": "user", "type": "message", "content": "hi"}])
    with mock.patch(
        "interpreter.terminal_interface.magic_commands.count_messages_tokens",
        side_effect=[(10, 0.001), (5, 0.0005)],
    ):
        magic_commands.handle_count_tokens(interpreter, "extra prompt")
    text = interpreter.display_message.call_args[0][0]
    assert "Total tokens for next request" in text


def test_handle_help_lists_commands():
    """%help output lists available magic commands such as %undo and %jupyter."""
    interpreter = _interpreter()
    magic_commands.handle_help(interpreter, "")
    text = interpreter.display_message.call_args[0][0]
    assert "%undo" in text
    assert "%jupyter" in text


def test_handle_reset_displays_confirmation():
    """%reset shows a confirmation message after clearing the session."""
    interpreter = _interpreter()
    magic_commands.handle_reset(interpreter, "")
    interpreter.reset.assert_called_once()
    assert "Reset Done" in interpreter.display_message.call_args[0][0]


def test_handle_undo_shows_function_call_preview():
    """%undo after a function call removes the block and reports the removal."""
    interpreter = _interpreter(
        messages=[
            {"role": "user", "content": "run it"},
            {"role": "assistant", "function_call": {"name": "execute"}},
        ]
    )
    magic_commands.handle_undo(interpreter, "")
    assert interpreter.messages == []
    interpreter.display_message.assert_called_with("**Removed codeblock**")


def test_handle_verbose_unknown_argument():
    """%verbose with an invalid argument shows an unknown-argument error."""
    interpreter = _interpreter()
    magic_commands.handle_verbose(interpreter, "maybe")
    assert "Unknown argument" in interpreter.display_message.call_args[0][0]


def test_handle_debug_toggles_flag():
    """%debug true/false toggles interpreter.debug."""
    interpreter = _interpreter(debug=False)
    magic_commands.handle_debug(interpreter, "true")
    assert interpreter.debug is True
    magic_commands.handle_debug(interpreter, "false")
    assert interpreter.debug is False


def test_default_handle_shows_help():
    """Unknown magic commands trigger default_handle, which warns and shows help."""
    interpreter = _interpreter()
    magic_commands.default_handle(interpreter, "")
    calls = [c[0][0] for c in interpreter.display_message.call_args_list]
    assert any("Unknown command" in c for c in calls)
    assert any("Available Commands" in c for c in calls)


def test_handle_info_delegates_to_system_info():
    """%info delegates to system_info with the interpreter instance."""
    interpreter = _interpreter()
    with mock.patch("interpreter.terminal_interface.magic_commands.system_info") as info:
        magic_commands.handle_info(interpreter, "")
    info.assert_called_once_with(interpreter)


def test_handle_save_message_adds_json_extension(tmp_path):
    """%save_message appends .json when the path has no extension."""
    interpreter = _interpreter(messages=[{"role": "user", "content": "x"}])
    path = tmp_path / "out"
    magic_commands.handle_save_message(interpreter, str(path))
    assert path.with_suffix(".json").exists()


def test_handle_magic_command_runs_shell():
    """%% prefix runs the remainder as a shell command via computer.run."""
    interpreter = _interpreter()
    interpreter.computer = SimpleNamespace(run=mock.Mock())
    magic_commands.handle_magic_command(interpreter, "%% ls -la")
    interpreter.computer.run.assert_called_once_with(
        "shell", "ls -la", stream=False, display=True
    )


def test_handle_magic_command_debug_redirects_to_verbose():
    """%debug is deprecated and redirects to handle_verbose after a short delay."""
    interpreter = _interpreter(verbose=False)
    with mock.patch.object(magic_commands.time, "sleep") as sleep:
        with mock.patch.object(magic_commands, "handle_verbose") as handle_verbose:
            magic_commands.handle_magic_command(interpreter, "%debug true")
    sleep.assert_called_once_with(1.5)
    handle_verbose.assert_called_once_with(interpreter, "true")


def test_handle_magic_command_unknown_invokes_default():
    """Unrecognized % commands fall through to default_handle."""
    interpreter = _interpreter()
    with mock.patch.object(magic_commands, "default_handle") as default_handle:
        magic_commands.handle_magic_command(interpreter, "%nope")
    default_handle.assert_called_once_with(interpreter, "")


def test_markdown_export_empty_messages(capsys):
    """%markdown with no messages prints a no-messages-to-export notice."""
    interpreter = _interpreter(messages=[])
    magic_commands.markdown(interpreter, "")
    assert "No messages to export" in capsys.readouterr().out


def test_markdown_export_delegates(tmp_path):
    """%markdown with a path delegates to export_to_markdown."""
    interpreter = _interpreter(
        messages=[{"role": "user", "content": "hi"}],
        conversation_filename="chat.json",
    )
    export_path = str(tmp_path / "out.md")
    with mock.patch(
        "interpreter.terminal_interface.magic_commands.export_to_markdown"
    ) as export:
        magic_commands.markdown(interpreter, export_path)
    export.assert_called_once_with(interpreter.messages, export_path)


def test_jupyter_exports_notebook(tmp_path):
    """%jupyter builds a notebook from messages and writes it to Downloads."""
    import types

    interpreter = _interpreter(
        messages=[
            {"role": "user", "type": "message", "content": "question"},
            {"role": "assistant", "type": "code", "format": "python", "content": "print(1)"},
        ]
    )
    captured = {}

    v4_module = types.ModuleType("nbformat.v4")
    v4_module.new_markdown_cell = lambda content: {"cell_type": "markdown", "source": content}
    v4_module.new_code_cell = lambda content: mock.Mock(metadata={}, source=content)
    v4_module.new_notebook = lambda: captured.setdefault("nb", {"cells": []})

    nbformat_module = types.ModuleType("nbformat")
    nbformat_module.write = mock.Mock()
    nbformat_module.v4 = v4_module

    with mock.patch.object(
        magic_commands, "install_and_import", return_value=nbformat_module
    ):
        with mock.patch.object(
            magic_commands, "get_downloads_path", return_value=str(tmp_path)
        ):
            with mock.patch.dict(
                "sys.modules",
                {"nbformat": nbformat_module, "nbformat.v4": v4_module},
            ):
                magic_commands.jupyter(interpreter, "")

    cells = captured["nb"]["cells"]
    assert len(cells) == 2
    assert cells[0]["source"].startswith("> question")
    assert cells[1].metadata == {"language": "python"}
    nbformat_module.write.assert_called_once()
    interpreter.display_message.assert_called()


def test_install_and_import_returns_existing_module():
    """Already-imported packages are returned without calling pip."""
    fake = object()
    with mock.patch.dict("sys.modules", {"already_there": fake}):
        result = magic_commands.install_and_import("already_there")
    assert result is fake


def test_install_and_import_installs_via_pip_then_imports():
    """Missing packages are pip-installed, then re-imported and returned."""
    fake = object()
    with mock.patch(
        "builtins.__import__",
        side_effect=[ImportError("missing"), fake],
    ):
        with mock.patch.object(
            magic_commands.subprocess, "check_call"
        ) as check_call:
            result = magic_commands.install_and_import("somedummy")

    check_call.assert_called_once_with(
        [sys.executable, "-m", "pip", "install", "somedummy"],
        stdout=magic_commands.subprocess.DEVNULL,
        stderr=magic_commands.subprocess.DEVNULL,
    )
    assert result is fake


def test_install_and_import_pip_failure_unbound_module_known_bug():
    """KNOWN BUG: when pip fails and pip3 also fails, install_and_import
    raises UnboundLocalError instead of returning None. The function's
    finally block references `module`, which is never bound on the failure
    paths. Documenting current behavior."""
    with mock.patch(
        "builtins.__import__", side_effect=ImportError("missing")
    ):
        with mock.patch.object(
            magic_commands.subprocess,
            "check_call",
            side_effect=[
                magic_commands.subprocess.CalledProcessError(1, "pip"),
                magic_commands.subprocess.CalledProcessError(1, "pip3"),
            ],
        ):
            with pytest.raises(UnboundLocalError):
                magic_commands.install_and_import("somedummy")


def test_handle_undo_previews_removed_message_content():
    """%undo prints a preview of each removed message's content."""
    interpreter = _interpreter(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": "this is the reply"},
        ]
    )
    magic_commands.handle_undo(interpreter, "")
    assert interpreter.messages == [{"role": "user", "content": "first"}]
    # One preview per removed message, in order.
    assert interpreter.display_message.call_count == 2
    previews = [c[0][0] for c in interpreter.display_message.call_args_list]
    assert "run it" in previews[0]
    assert "this is the reply" in previews[1]


def test_handle_verbose_truncates_inline_images(capsys):
    """%verbose truncates non-path inline image content before printing."""
    interpreter = _interpreter(
        messages=[
            {
                "role": "user",
                "type": "image",
                "format": "base64",
                "content": "A" * 100,
            }
        ]
    )
    magic_commands.handle_verbose(interpreter, "true")
    assert interpreter.verbose is True
    printed = capsys.readouterr().out
    assert "..." in printed
    assert "A" * 100 not in printed  # the inline content was truncated


def test_handle_debug_truncates_inline_images(capsys):
    """%debug truncates non-path inline image content before printing."""
    interpreter = _interpreter(
        messages=[
            {
                "role": "user",
                "type": "image",
                "format": "data-url",
                "content": "B" * 100,
            }
        ]
    )
    magic_commands.handle_debug(interpreter, "true")
    assert interpreter.debug is True
    printed = capsys.readouterr().out
    assert "..." in printed
    assert "B" * 100 not in printed  # the inline content was truncated


def test_markdown_default_path_uses_downloads(monkeypatch, tmp_path):
    """%markdown without a path exports to Downloads using conversation name."""
    interpreter = _interpreter(
        messages=[{"role": "user", "content": "hi"}],
        conversation_filename="chat.json",
    )
    monkeypatch.setattr(
        magic_commands, "get_downloads_path", lambda: str(tmp_path)
    )
    with mock.patch(
        "interpreter.terminal_interface.magic_commands.export_to_markdown"
    ) as export:
        magic_commands.markdown(interpreter, "")

    export.assert_called_once_with(
        interpreter.messages, f"{tmp_path}/chat.md"
    )


def test_jupyter_handles_assistant_markdown_and_default_language(tmp_path):
    """%jupyter renders assistant messages as markdown and defaults code
    cells without a format to python."""
    import types

    interpreter = _interpreter(
        messages=[
            {"role": "assistant", "type": "message", "content": "assistant note"},
            {"role": "assistant", "type": "code", "content": "print(2)"},
        ]
    )
    captured = {}

    v4_module = types.ModuleType("nbformat.v4")
    v4_module.new_markdown_cell = lambda content: {
        "cell_type": "markdown",
        "source": content,
    }
    v4_module.new_code_cell = lambda content: mock.Mock(metadata={}, source=content)
    v4_module.new_notebook = lambda: captured.setdefault("nb", {"cells": []})

    nbformat_module = types.ModuleType("nbformat")
    nbformat_module.write = mock.Mock()
    nbformat_module.v4 = v4_module

    with mock.patch.object(
        magic_commands, "install_and_import", return_value=nbformat_module
    ):
        with mock.patch.object(
            magic_commands, "get_downloads_path", return_value=str(tmp_path)
        ):
            with mock.patch.dict(
                "sys.modules",
                {"nbformat": nbformat_module, "nbformat.v4": v4_module},
            ):
                magic_commands.jupyter(interpreter, "")

    cells = captured["nb"]["cells"]
    assert cells[0] == {"cell_type": "markdown", "source": "assistant note"}
    assert cells[1].metadata == {"language": "python"}
