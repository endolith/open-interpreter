from unittest import mock

import pytest

from interpreter.terminal_interface.terminal_interface import terminal_interface


def _make_interpreter():
    """Build an interpreter stub that avoids TTY/LLM/network side effects.

    auto_run + offline skip the intro message, and the vision/OS flags are
    disabled so the render loop stays on the plain message/code-block paths.
    """
    interpreter = mock.MagicMock()
    interpreter.auto_run = True
    interpreter.offline = True
    interpreter.messages = []
    interpreter.plain_text_display = False
    interpreter.os = False
    interpreter.safe_mode = "off"
    interpreter.verbose = False
    interpreter.multi_line = False
    interpreter.max_output = 2000
    interpreter.llm.supports_vision = False
    interpreter.llm.vision_renderer = None
    return interpreter


def test_terminal_interface_yields_chunks_and_renders_message_block():
    """terminal_interface yields each chat chunk and renders an assistant message block."""
    interpreter = _make_interpreter()

    def chat(message, display=False, stream=True):
        yield {"type": "message", "role": "assistant", "start": True}
        yield {"type": "message", "role": "assistant", "content": "hello"}
        yield {"type": "message", "role": "assistant", "end": True}

    interpreter.chat = chat

    chunks = list(terminal_interface(interpreter, "say hi"))

    assert chunks == [
        {"type": "message", "role": "assistant", "start": True},
        {"type": "message", "role": "assistant", "content": "hello"},
        {"type": "message", "role": "assistant", "end": True},
    ]


def test_terminal_interface_yields_chunks_and_renders_code_block():
    """terminal_interface yields each chat chunk and renders an assistant code block."""
    interpreter = _make_interpreter()

    def chat(message, display=False, stream=True):
        yield {"type": "code", "role": "assistant", "start": True, "format": "python"}
        yield {"type": "code", "role": "assistant", "content": "x = 1\n"}
        yield {"type": "code", "role": "assistant", "end": True}

    interpreter.chat = chat

    chunks = list(terminal_interface(interpreter, "write code"))

    assert len(chunks) == 3
    assert chunks[0]["type"] == "code"
    assert chunks[-1]["type"] == "code"


def _intro_interpreter():
    """An interpreter that shows the interactive intro message (needs approval)."""
    interpreter = _make_interpreter()
    interpreter.auto_run = False
    interpreter.offline = False
    return interpreter


def test_terminal_interface_shows_approval_intro():
    """terminal_interface tells interactive users code needs approval when
    auto_run is off."""
    interpreter = _intro_interpreter()
    interpreter.chat = lambda message, display=False, stream=True: iter([])

    list(terminal_interface(interpreter, "hello"))

    intro = interpreter.display_message.call_args[0][0]
    assert "will require approval before running code" in intro
    assert "Use `interpreter -y` to bypass this." in intro


def test_terminal_interface_skips_intro_when_auto_run():
    """terminal_interface does not show the approval intro when auto_run is on."""
    interpreter = _make_interpreter()
    interpreter.chat = lambda message, display=False, stream=True: iter([])

    list(terminal_interface(interpreter, "hello"))

    interpreter.display_message.assert_not_called()


def test_terminal_interface_safe_mode_ask_mentions_semgrep():
    """terminal_interface notes that ask/auto safe mode needs semgrep."""
    import interpreter.terminal_interface.terminal_interface as ti

    interpreter = _intro_interpreter()
    interpreter.safe_mode = "ask"
    interpreter.chat = lambda message, display=False, stream=True: iter([])
    with mock.patch.object(ti, "check_for_package", return_value=False):
        list(terminal_interface(interpreter, "hello"))

    intro = interpreter.display_message.call_args[0][0]
    assert "Safe Mode" in intro
    assert "semgrep" in intro


def test_terminal_interface_ignores_empty_input():
    """terminal_interface skips an empty line instead of sending it to the LLM."""
    interpreter = _intro_interpreter()

    with mock.patch("builtins.input", side_effect=["", KeyboardInterrupt()]):
        with pytest.raises(KeyboardInterrupt):
            list(terminal_interface(interpreter, ""))

    assert "Exiting..." in interpreter.display_message.call_args[0][0]
    interpreter.chat.assert_not_called()


def test_terminal_interface_dispatches_magic_command():
    """A message starting with % is handed to handle_magic_command."""
    import interpreter.terminal_interface.terminal_interface as ti

    interpreter = _intro_interpreter()
    with mock.patch.object(ti, "handle_magic_command") as handle:
        with mock.patch("builtins.input", side_effect=["%help", KeyboardInterrupt()]):
            with pytest.raises(KeyboardInterrupt):
                list(terminal_interface(interpreter, ""))

    handle.assert_called_once_with(interpreter, "%help")
    interpreter.chat.assert_not_called()


def test_terminal_interface_local_command_hint(capsys):
    """terminal_interface points `interpreter --local` users back to the CLI."""
    interpreter = _intro_interpreter()
    with mock.patch(
        "builtins.input",
        side_effect=["interpreter --local", KeyboardInterrupt()],
    ):
        with pytest.raises(KeyboardInterrupt):
            list(terminal_interface(interpreter, ""))

    assert "Please exit this conversation" in capsys.readouterr().out
    interpreter.chat.assert_not_called()


def test_terminal_interface_convert_image_path_to_image_message():
    """A dragged-in image path is turned into an image message for the LLM."""
    import interpreter.terminal_interface.terminal_interface as ti

    interpreter = _make_interpreter()
    interpreter.llm.supports_vision = True
    with mock.patch.object(ti, "find_image_path", return_value="/tmp/pic.png") as find:
        list(terminal_interface(interpreter, "an image"))

    find.assert_called_once_with("an image")
    assert interpreter.messages == [
        {"role": "user", "type": "message", "content": "an image"}
    ]
    interpreter.chat.assert_called_once_with(
        {"role": "user", "type": "image", "format": "path", "content": "/tmp/pic.png"},
        display=False,
        stream=True,
    )


def test_terminal_interface_plain_text_renders_code_fences(capsys):
    """In plain-text mode, code chunks are printed inside markdown fences."""
    interpreter = _make_interpreter()
    interpreter.plain_text_display = True

    def chat(message, display=False, stream=True):
        yield {"type": "code", "role": "assistant", "format": "python", "start": True}
        yield {"type": "code", "role": "assistant", "content": "print(1)"}
        yield {"type": "code", "role": "assistant", "format": "python", "end": True}

    interpreter.chat = chat

    list(terminal_interface(interpreter, "hi"))

    output = capsys.readouterr().out
    assert "```python" in output
    assert "print(1)" in output
    assert "```" in output


def test_terminal_interface_os_failsafe_stops_loop(capsys):
    """A PyAutoGUI failsafe output in OS mode stops the current run."""
    interpreter = _make_interpreter()
    interpreter.os = True
    interpreter.chat = lambda message, display=False, stream=True: iter(
        [{"type": "console", "format": "output", "content": "FailSafeException"}]
    )

    list(terminal_interface(interpreter, "hi"))

    assert "Fail-safe triggered" in capsys.readouterr().out


def test_terminal_interface_prints_review_chunk(capsys):
    """terminal_interface prints code-review chunks as they stream."""
    interpreter = _make_interpreter()
    interpreter.chat = lambda message, display=False, stream=True: iter(
        [{"type": "review", "role": "assistant", "content": "LGTM"}]
    )

    list(terminal_interface(interpreter, "review"))

    assert "LGTM" in capsys.readouterr().out


def test_terminal_interface_declined_code_is_recorded():
    """Declining a confirmation records the refusal in the message history."""
    interpreter = _make_interpreter()
    interpreter.auto_run = False

    def chat(message, display=False, stream=True):
        yield {
            "type": "confirmation",
            "content": {"format": "python", "content": "print(1)"},
        }

    interpreter.chat = chat
    with mock.patch("builtins.input", return_value="n"):
        list(terminal_interface(interpreter, "run code"))

    assert interpreter.messages[-1]["content"] == "I have declined to run this code."


def test_terminal_interface_os_mode_notifies_on_message_end():
    """In OS mode, the end of an assistant message triggers an OS notification."""
    import interpreter.terminal_interface.terminal_interface as ti

    interpreter = _make_interpreter()
    interpreter.os = True
    interpreter.messages = [{"role": "assistant", "content": "- item one\nline two"}]

    def chat(message, display=False, stream=True):
        yield {"type": "message", "role": "assistant", "start": True}
        yield {"type": "message", "role": "assistant", "content": "hello"}
        yield {"type": "message", "role": "assistant", "end": True}

    interpreter.chat = chat
    with mock.patch.object(ti.platform, "system", return_value="Linux"):
        list(terminal_interface(interpreter, "notify"))

    # The markdown list line and the line above it are stripped before notifying.
    interpreter.computer.os.notify.assert_called_once_with("line two")
