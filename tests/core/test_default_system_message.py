import getpass
import os
import platform

from interpreter import OpenInterpreter
from interpreter.core.default_system_message import build_default_system_message
from interpreter.core.respond import build_system_message, respond


def test_default_system_message_is_nonempty_string():
    """The builder returns a non-empty string instructing the model."""
    message = build_default_system_message()
    assert isinstance(message, str)
    assert message.strip()


def test_default_system_message_identifies_open_interpreter():
    """The system message declares the assistant as Open Interpreter."""
    assert "Open Interpreter" in build_default_system_message()


def test_default_system_message_includes_user_context():
    """The system message embeds the user's name and OS."""
    message = build_default_system_message()
    assert getpass.getuser() in message
    assert platform.system() in message


def test_default_system_message_includes_cwd():
    """The model is told the directory it was started in, so relative paths resolve correctly."""
    assert os.getcwd() in build_default_system_message()


def test_interpreter_captures_cwd_at_initialization(monkeypatch, tmp_path):
    """Changing directory after import must be reflected in a new session's message."""
    monkeypatch.chdir(tmp_path)

    interpreter = OpenInterpreter(offline=True, disable_telemetry=True)

    assert str(tmp_path) in interpreter.system_message


def test_default_system_message_refreshes_cwd_for_each_turn(monkeypatch, tmp_path):
    """A directory change after the interpreter was created is reflected in the turn."""
    interpreter = OpenInterpreter(offline=True, disable_telemetry=True)
    monkeypatch.chdir(tmp_path)

    assert str(tmp_path) in build_system_message(interpreter)


def test_explicit_system_message_is_preserved(monkeypatch, tmp_path):
    """A caller-supplied system message is neither replaced nor rebuilt."""
    interpreter = OpenInterpreter(
        offline=True, disable_telemetry=True, system_message="Be brief."
    )
    monkeypatch.chdir(tmp_path)

    assert build_system_message(interpreter) == "Be brief."


def test_working_directory_is_not_rendered_as_a_template(monkeypatch, tmp_path):
    """A directory name containing template syntax is data, not executable code."""
    dangerous = tmp_path / "{{open('pwned.txt', 'w').write('x')}}"
    dangerous.mkdir()
    monkeypatch.chdir(dangerous)

    interpreter = OpenInterpreter(offline=True, disable_telemetry=True)
    interpreter.system_message += "\n\nBe brief."
    message = build_system_message(interpreter)

    assert str(dangerous) in message
    assert message.endswith("Be brief.")
    assert not (dangerous / "pwned.txt").exists()


def test_respond_sends_the_fresh_default_message(monkeypatch, tmp_path):
    """respond() sends the rebuilt default message rather than the import-time copy."""
    interpreter = OpenInterpreter(offline=True, disable_telemetry=True)
    interpreter.messages = [
        {"role": "user", "type": "message", "content": "hello"}
    ]
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_run(messages, **kwargs):
        captured["system"] = messages[0]["content"]
        return iter([{"type": "message", "content": "ok"}])

    interpreter.llm.run = fake_run
    stream = respond(interpreter)
    next(stream)
    stream.close()

    assert str(tmp_path) in captured["system"]


def test_caller_addition_keeps_the_default_message_refreshable(monkeypatch, tmp_path):
    """`system_message += ...` must not pin the directory captured at import."""
    interpreter = OpenInterpreter(offline=True, disable_telemetry=True)
    interpreter.system_message += "\n\nAlways answer in French."
    monkeypatch.chdir(tmp_path)

    message = build_system_message(interpreter)

    assert str(tmp_path) in message
    assert message.endswith("Always answer in French.")


def test_replacement_after_init_is_used_verbatim(monkeypatch, tmp_path):
    """Replacing the message after construction opts out of the default refresh."""
    interpreter = OpenInterpreter(offline=True, disable_telemetry=True)
    interpreter.system_message = "You are a terse assistant."
    monkeypatch.chdir(tmp_path)

    assert build_system_message(interpreter) == "You are a terse assistant."


def test_prepended_text_keeps_the_default_refreshable(monkeypatch, tmp_path):
    """Adding text before the generated block does not pin its directory."""
    interpreter = OpenInterpreter(offline=True, disable_telemetry=True)
    interpreter.system_message = "Intro.\n\n" + interpreter.system_message
    monkeypatch.chdir(tmp_path)

    message = build_system_message(interpreter)

    assert message.startswith("Intro.")
    assert str(tmp_path) in message


def test_edited_default_keeps_the_directory_literal(monkeypatch, tmp_path):
    """Editing the generated text must not turn its directory into a template."""
    dangerous = tmp_path / "{{open('pwned.txt', 'w').write('x')}}"
    dangerous.mkdir()
    monkeypatch.chdir(dangerous)

    interpreter = OpenInterpreter(offline=True, disable_telemetry=True)
    interpreter.system_message = interpreter.system_message.replace(
        "world-class", "excellent"
    )
    message = build_system_message(interpreter)

    assert "excellent" in message
    assert str(dangerous) in message
    assert not (dangerous / "pwned.txt").exists()


def test_unreadable_cwd_omits_the_directory_line(monkeypatch):
    """A deleted working directory must not break import or a response."""
    def raise_missing_directory():
        raise FileNotFoundError("working directory was removed")

    monkeypatch.setattr(
        "interpreter.core.default_system_message.os.getcwd",
        raise_missing_directory,
    )

    message = build_default_system_message()

    assert "Current working directory:" not in message
