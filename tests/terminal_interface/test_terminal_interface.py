from unittest import mock

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
