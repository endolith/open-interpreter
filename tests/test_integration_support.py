import pytest

from tests import integration_support as support


def test_integration_tests_allowed_requires_api_key(monkeypatch):
    """An opt-in without a provider key must not be treated as runnable."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OI_RUN_INTEGRATION", "1")
    assert support.integration_tests_allowed() is False


def test_integration_tests_allowed_requires_local_opt_in(monkeypatch):
    """A key alone is not consent: local runs also need OI_RUN_INTEGRATION=1."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("OI_RUN_INTEGRATION", raising=False)
    assert support.integration_tests_allowed() is False


def test_integration_tests_allowed_on_ci(monkeypatch):
    """CI runners are disposable, so GITHUB_ACTIONS=true implies consent."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("OI_RUN_INTEGRATION", raising=False)
    assert support.integration_tests_allowed() is True


def test_prompt_for_code_execution_can_approve_all(monkeypatch):
    """Answering "a" approves this block and every later one without prompting."""
    monkeypatch.setattr(support, "_approve_all", False)
    monkeypatch.setattr(support, "_approved_hashes", set())
    monkeypatch.delenv("OI_AUTO_APPROVE_INTEGRATION", raising=False)
    monkeypatch.setattr(support, "_terminal_stream", lambda: object())
    responses = iter(["a"])
    monkeypatch.setattr(support, "_tty_input", lambda _prompt: next(responses))
    monkeypatch.setattr(support, "_tty_print", lambda _msg: None)
    assert support.prompt_for_code_execution(test_name="demo", language="python", code='print("hi")')
    assert support.prompt_for_code_execution(test_name="demo", language="python", code='print("other")')


def test_install_chat_approval_blocks_when_denied(monkeypatch):
    """A denied confirmation chunk fails the test instead of executing the code."""
    monkeypatch.setattr(support, "_approve_all", False)
    monkeypatch.setattr(support, "_approved_hashes", set())
    monkeypatch.setattr(support, "prompt_for_code_execution", lambda **_: False)

    from interpreter import OpenInterpreter
    from interpreter.core import core as core_mod

    confirmation_chunk = {
        "role": "computer",
        "type": "confirmation",
        "format": "execution",
        "content": {"type": "code", "format": "python", "content": 'print("hi")'},
    }

    def fake_chat(self, message=None, display=False, stream=False, blocking=True):
        yield dict(confirmation_chunk)

    # Install over the fake so the wrapper's captured original chats like a real
    # response that wants to execute code.
    monkeypatch.setattr(core_mod.OpenInterpreter, "chat", fake_chat)
    support.install_chat_approval(monkeypatch, "demo_test")

    oi = OpenInterpreter()
    with pytest.raises(pytest.fail.Exception, match="not approved"):
        list(oi.chat("test", display=False, stream=True))


def test_prompt_fails_without_a_terminal(monkeypatch):
    """A session with no controlling terminal fails loudly instead of hanging.

    fd capture redirects fds 0-2, so the prompt must open the terminal itself;
    when that is impossible (CI, piped output) the test fails with a message
    naming the opt-out rather than blocking on a read that can never succeed.
    """
    monkeypatch.setattr(support, "_approve_all", False)
    monkeypatch.setattr(support, "_approved_hashes", set())
    monkeypatch.setattr(support, "_terminal_stream", lambda: None)
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
    monkeypatch.setattr(support, "_terminal_stream", lambda: terminal)
    monkeypatch.delenv("OI_AUTO_APPROVE_INTEGRATION", raising=False)
    assert support.prompt_for_code_execution(test_name="demo", language="python", code='print("hi")')
    rendered = "".join(terminal.written)
    assert "Run this code?" in rendered
    assert "print(\"hi\")" in rendered


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


def test_chat_wrapper_non_streaming_returns_stored_messages(monkeypatch):
    """The non-streaming branch mirrors chat(stream=False) instead of rebuilding chunks.

    It drains the guarded stream, resets responding, and returns the slice of
    stored messages _streaming_chat produced.
    """
    from interpreter import OpenInterpreter
    from interpreter.core import core as core_mod

    def fake_chat(self, message=None, display=True, stream=False, blocking=True):
        yield {"role": "assistant", "type": "message", "content": "hello"}

    monkeypatch.setattr(core_mod.OpenInterpreter, "chat", fake_chat)
    support.install_chat_approval(monkeypatch, "demo")
    oi = OpenInterpreter()
    oi.messages = [
        {"role": "user", "type": "message", "content": "hi"},
        {"role": "assistant", "type": "message", "content": "hello"},
    ]
    oi.last_messages_count = 1
    oi.responding = True
    assert oi.chat("hi", display=False) == [
        {"role": "assistant", "type": "message", "content": "hello"}
    ]
    assert oi.responding is False
