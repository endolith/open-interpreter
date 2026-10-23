import pytest

from interpreter.terminal_interface.terminal_interface import (
    NO_APPROVER_NOTICE,
    terminal_interface,
)


class _FakeLLM:
    supports_vision = False
    vision_renderer = None


class _FakeInterpreter:
    """Just enough of an Interpreter for one pass of terminal_interface."""

    verbose = False
    offline = False
    os = None
    plain_text_display = True
    highlight_active_line = False
    safe_mode = "off"
    auto_run_mode = "prompt"

    def __init__(self, chunks):
        self.llm = _FakeLLM()
        self._chunks = chunks
        self.messages = [{"role": "user", "type": "message", "content": "hi"}]

    def chat(self, message, display=False, stream=True):
        yield from self._chunks

    def display_message(self, text):
        pass


_RUN = {
    "role": "computer",
    "type": "confirmation",
    "format": "execution",
    "content": {"type": "code", "format": "python", "content": "print(1)"},
}
_EDIT = {
    "role": "computer",
    "type": "confirmation",
    "format": "edit",
    "content": {"format": "write", "content": "{}", "target": "/x/y.txt"},
}


@pytest.mark.parametrize("chunk", [_RUN, _EDIT], ids=["run", "edit"])
def test_headless_rejection_is_reported_on_the_console(no_tty, capsys, chunk):
    """A step rejected for want of an approver says so where the operator looks.

    With no TTY, prompt_choice raises before it prints its question, and the
    model-facing NO_APPROVER_NOTICE only goes into the message history. Without
    a console line, a headless run shows the proposed code and then simply
    ends, with nothing telling the operator why it did not run or that -y
    would approve it.
    """
    interp = _FakeInterpreter([chunk])
    list(terminal_interface(interp, "do it"))

    out = capsys.readouterr().out
    assert "no interactive terminal" in out
    assert "-y" in out
    # The model-facing notice is still recorded for the next turn.
    assert interp.messages[-1]["content"] == NO_APPROVER_NOTICE
