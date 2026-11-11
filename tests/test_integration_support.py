import pytest

from tests import integration_support as support


def test_prompt_for_code_execution_can_approve_all(monkeypatch):
    """Answering "a" approves this block and every later one without prompting."""
    monkeypatch.setattr(support, "_approve_all", False)
    monkeypatch.setattr(support, "_approved_hashes", set())
    monkeypatch.delenv("OI_AUTO_APPROVE_INTEGRATION", raising=False)
    monkeypatch.setattr(support, "_terminal_handles", lambda: (object(), object()))
    responses = iter(["a"])
    monkeypatch.setattr(support, "_tty_input", lambda _prompt: next(responses))
    monkeypatch.setattr(support, "_tty_print", lambda _msg: None)
    assert support.prompt_for_code_execution(test_name="demo", language="python", code='print("hi")')
    assert support.prompt_for_code_execution(test_name="demo", language="python", code='print("other")')


def test_install_approval_blocks_execution_even_with_auto_run(monkeypatch):
    """A denied block must not run just because ``auto_run`` is True.

    ``tests/test_interpreter.py`` sets ``interpreter.auto_run = True``, and
    ``core.py`` drops the confirmation chunk in that state, so a guard watching
    the chunk never saw it and the code executed with no prompt. The guard now
    sits at ``Computer.run``, the seam ``respond.py`` executes through.
    """
    from interpreter import OpenInterpreter
    from interpreter.core.computer.computer import Computer

    executed = []

    def recording_run(self, *args, **kwargs):
        executed.append((args, kwargs))

    monkeypatch.setattr(Computer, "run", recording_run)
    prompts = []

    def denying_prompt(**kwargs):
        prompts.append(kwargs)
        return False

    monkeypatch.setattr(support, "prompt_for_code_execution", denying_prompt)
    support.install_chat_approval(monkeypatch, "demo_test")

    oi = OpenInterpreter()
    oi.auto_run = True
    with pytest.raises(pytest.fail.Exception, match="not approved"):
        oi.computer.run("python", 'print("hi")', stream=True)

    assert prompts == [
        {"test_name": "demo_test", "language": "python", "code": 'print("hi")'}
    ]
    assert executed == []


def test_install_approval_runs_an_approved_block(monkeypatch):
    """An approved block reaches the terminal, and only once."""
    from interpreter import OpenInterpreter
    from interpreter.core.computer.computer import Computer

    executed = []

    def recording_run(self, *args, **kwargs):
        executed.append((args, kwargs))

    monkeypatch.setattr(Computer, "run", recording_run)
    monkeypatch.setattr(support, "prompt_for_code_execution", lambda **_: True)
    support.install_chat_approval(monkeypatch, "demo")

    oi = OpenInterpreter()
    oi.computer.run("python", 'print("hi")', stream=True)

    assert executed == [(("python", 'print("hi")'), {"stream": True})]


def test_install_approval_ignores_internal_runs(monkeypatch):
    """Scaffolding calls must not prompt; only the model's block carries stream=True.

    ``sync_computer`` and the computer-API import both call ``Computer.run``
    without ``stream``. Prompting for those would ask the user to approve code
    the interpreter wrote itself, and would fire several times per test.
    """
    from interpreter import OpenInterpreter
    from interpreter.core.computer.computer import Computer

    executed = []

    def recording_run(self, *args, **kwargs):
        executed.append((args, kwargs))

    monkeypatch.setattr(Computer, "run", recording_run)
    prompts = []
    monkeypatch.setattr(
        support,
        "prompt_for_code_execution",
        lambda **kwargs: prompts.append(kwargs) or True,
    )
    support.install_chat_approval(monkeypatch, "demo")

    oi = OpenInterpreter()
    oi.computer.run("python", "computer.load_dict({})")

    assert prompts == []
    assert executed == [(("python", "computer.load_dict({})"), {})]


def test_prompt_fails_without_a_terminal(monkeypatch):
    """A session with no controlling terminal fails loudly instead of hanging.

    fd capture redirects fds 0-2, so the prompt must open the terminal itself;
    when that is impossible (CI, piped output) the test fails with a message
    naming the opt-out rather than blocking on a read that can never succeed.
    """
    monkeypatch.setattr(support, "_approve_all", False)
    monkeypatch.setattr(support, "_approved_hashes", set())
    monkeypatch.setattr(support, "_terminal_handles", lambda: None)
    monkeypatch.delenv("OI_AUTO_APPROVE_INTEGRATION", raising=False)
    with pytest.raises(pytest.fail.Exception, match="no interactive terminal"):
        support.prompt_for_code_execution(test_name="demo", language="python", code='print("hi")')


def test_prompt_writes_and_reads_the_controlling_terminal(monkeypatch):
    """The code block and the answer travel over the terminal handle, not stdout.

    sys.__stderr__/sys.__stdin__ are captured under --capture=fd, so a prompt
    that used them would never reach the user (review finding on #578).
    """

    class FakeTerminal:
        def __init__(self):
            self.written = []
            self.answers = iter(["y\n"])

        def write(self, text):
            self.written.append(text)

        def flush(self):
            pass

        def readline(self):
            return next(self.answers)

    terminal = FakeTerminal()
    monkeypatch.setattr(support, "_approve_all", False)
    monkeypatch.setattr(support, "_approved_hashes", set())
    monkeypatch.setattr(support, "_terminal_handles", lambda: (terminal, terminal))
    monkeypatch.delenv("OI_AUTO_APPROVE_INTEGRATION", raising=False)
    assert support.prompt_for_code_execution(test_name="demo", language="python", code='print("hi")')
    rendered = "".join(terminal.written)
    assert "Run this code?" in rendered
    assert "print(\"hi\")" in rendered


def test_terminal_handles_open_read_and_write_separately(monkeypatch):
    """Input and output get separate handles, because a tty cannot be opened "r+".

    Text-mode update mode seeks the stream back to its start, and a terminal is
    not seekable: `open("/dev/tty", "r+")` raises io.UnsupportedOperation. A
    single "r+" handle therefore reported no terminal and silently disabled the
    approval prompt even in a real session.
    """
    modes = []

    class Handle:
        def __init__(self, mode):
            self.mode = mode

    def fake_open(mode):
        modes.append(mode)
        return Handle(mode)

    monkeypatch.setattr(support, "_terminal", None)
    monkeypatch.setattr(support, "_open_terminal", fake_open)
    reader, writer = support._terminal_handles()
    assert modes == ["r", "w"]
    assert (reader.mode, writer.mode) == ("r", "w")


def test_terminal_handles_degrade_to_none_when_output_cannot_open(monkeypatch):
    """A terminal that cannot be opened for writing counts as unavailable.

    The reader opened first must be closed rather than leaked, and the failure
    must be cached so every prompt does not retry the open.
    """
    closed = []

    class Handle:
        def close(self):
            closed.append(True)

    def fake_open(mode):
        if mode == "w":
            raise OSError("no terminal")
        return Handle()

    monkeypatch.setattr(support, "_terminal", None)
    monkeypatch.setattr(support, "_open_terminal", fake_open)
    assert support._terminal_handles() is None
    assert closed == [True]
    assert support._terminal is False


def test_chat_wrapper_keeps_the_display_default(monkeypatch):
    """A plain interpreter.chat() must still route through terminal_interface.

    The original signature defaults display=True; the wrapper used to force
    display=False and stream=True, silently bypassing terminal_interface for
    every default call.
    """
    from interpreter import OpenInterpreter
    from interpreter.core import core as core_mod

    calls = []

    def fake_chat(self, message=None, display=True, stream=False, blocking=True):
        calls.append({"display": display, "stream": stream, "blocking": blocking})
        return iter([{"role": "assistant", "type": "message", "content": "ok"}])

    monkeypatch.setattr(core_mod.OpenInterpreter, "chat", fake_chat)
    support.install_chat_approval(monkeypatch, "demo")
    OpenInterpreter().chat("hi")
    assert calls == [{"display": True, "stream": False, "blocking": True}]


def test_chat_wrapper_delegates_non_blocking_calls(monkeypatch):
    """blocking=False must reach the original, which starts the worker thread.

    The wrapper's guarded generator is unconsumed for blocking=False, so
    consuming raw there would break the call the original returns None for.
    """
    from interpreter import OpenInterpreter
    from interpreter.core import core as core_mod

    calls = []

    def fake_chat(self, message=None, display=True, stream=False, blocking=True):
        calls.append({"display": display, "stream": stream, "blocking": blocking})
        return None

    monkeypatch.setattr(core_mod.OpenInterpreter, "chat", fake_chat)
    support.install_chat_approval(monkeypatch, "demo")
    OpenInterpreter().chat("hi", display=False, blocking=False)
    assert calls == [{"display": False, "stream": False, "blocking": False}]


def test_chat_wrapper_pins_auto_run_and_restores_it(monkeypatch):
    """The execution guard owns the only prompt, so chat runs with auto_run pinned.

    Under display=True terminal_interface receives the confirmation chunk when
    ``auto_run`` is False and calls ``input()`` for it - under pytest's capture
    that read goes nowhere. Pinning it True during the chat keeps the terminal
    from asking a second time, and the original value must come back afterwards.
    """
    from interpreter import OpenInterpreter
    from interpreter.core import core as core_mod

    seen = []

    def fake_chat(self, message=None, display=True, stream=False, blocking=True):
        seen.append(self.auto_run)
        yield {"role": "assistant", "type": "message", "content": "hello"}

    monkeypatch.setattr(core_mod.OpenInterpreter, "chat", fake_chat)
    support.install_chat_approval(monkeypatch, "demo")
    oi = OpenInterpreter()
    oi.auto_run = False

    assert list(oi.chat("hi", display=False, stream=True)) == [
        {"role": "assistant", "type": "message", "content": "hello"}
    ]
    assert seen == [True]
    assert oi.auto_run is False


def test_chat_wrapper_returns_the_non_streaming_result_unchanged(monkeypatch):
    """Non-streaming calls delegate wholesale; the wrapper no longer rebuilds messages.

    ``chat(stream=False)`` already drains the stream and returns the new
    messages, so re-deriving them here would duplicate that logic and could
    drift from it.
    """
    from interpreter import OpenInterpreter
    from interpreter.core import core as core_mod

    sentinel = [{"role": "assistant", "type": "message", "content": "hello"}]

    def fake_chat(self, message=None, display=True, stream=False, blocking=True):
        return sentinel

    monkeypatch.setattr(core_mod.OpenInterpreter, "chat", fake_chat)
    support.install_chat_approval(monkeypatch, "demo")
    oi = OpenInterpreter()

    assert oi.chat("hi", display=False) is sentinel
    assert oi.auto_run is False


def test_chat_wrapper_restores_auto_run_when_chat_raises(monkeypatch):
    """A chat that raises must not leave the interpreter's auto_run flipped."""
    from interpreter import OpenInterpreter
    from interpreter.core import core as core_mod

    def failing_chat(self, message=None, display=True, stream=False, blocking=True):
        raise RuntimeError("boom")

    monkeypatch.setattr(core_mod.OpenInterpreter, "chat", failing_chat)
    support.install_chat_approval(monkeypatch, "demo")
    oi = OpenInterpreter()
    oi.auto_run = False

    with pytest.raises(RuntimeError, match="boom"):
        oi.chat("hi", display=False, stream=True)
    assert oi.auto_run is False
