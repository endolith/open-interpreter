from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.computer.terminal.terminal import (
    _SHELL_OUTPUT_FLUSH_AT,
    Terminal,
)


def test_get_language_by_name():
    """Terminal.get_language() resolves known language names and returns None for unknown ones."""
    terminal = Terminal(computer=SimpleNamespace())
    assert terminal.get_language("python").name == "Python"
    assert terminal.get_language("bash").name == "Shell"
    assert terminal.get_language("unknown_xyz") is None


def test_get_language_by_alias():
    """Terminal.get_language() resolves shell aliases such as 'sh' to Shell."""
    terminal = Terminal(computer=SimpleNamespace())
    assert terminal.get_language("sh").name == "Shell"


class _FakeShell:
    """Minimal language stub with an alias, for process-sharing tests."""

    name = "Shell"
    aliases = ["bash", "sh"]

    def __init__(self, computer):
        self.computer = computer

    def run(self, code):
        yield {"type": "console", "format": "output", "content": code}


def _bare_computer():
    return SimpleNamespace(
        import_computer_api=False,
        import_skills=False,
        _has_imported_computer_api=False,
        _has_imported_skills=False,
        verbose=False,
        skills=SimpleNamespace(import_skills=mock.Mock()),
    )


def test_canonical_language_resolves_aliases_and_passes_unknown_through():
    """canonical_language() maps aliases to the primary name, lowercased (issue #323)."""
    terminal = Terminal(computer=SimpleNamespace())
    terminal.languages = [_FakeShell]
    assert terminal.canonical_language("bash") == "shell"
    assert terminal.canonical_language("Shell") == "shell"
    assert terminal.canonical_language("unknown_xyz") == "unknown_xyz"


def test_run_alias_shares_process_with_canonical_name():
    """'bash' and 'shell' blocks share one persistent process (issue #323)."""
    terminal = Terminal(computer=_bare_computer())
    terminal.languages = [_FakeShell]
    list(terminal.run("bash", "cd /tmp", stream=True))
    list(terminal.run("shell", "pwd", stream=True))
    assert list(terminal._active_languages) == ["shell"]


def test_run_non_streaming_merges_output_chunks():
    """Terminal.run(stream=False) concatenates consecutive console output chunks."""
    computer = SimpleNamespace(
        import_computer_api=False,
        import_skills=False,
        _has_imported_computer_api=False,
        _has_imported_skills=False,
        verbose=False,
        skills=SimpleNamespace(import_skills=mock.Mock()),
    )
    terminal = Terminal(computer=computer)

    def fake_streaming_run(language, code, display=False):
        yield {"type": "console", "format": "output", "content": "part1"}
        yield {"type": "console", "format": "output", "content": "part2"}

    with mock.patch.object(terminal, "_streaming_run", side_effect=fake_streaming_run):
        output = terminal.run("fake", "code", stream=False)
    assert output == [
        {"type": "console", "format": "output", "content": "part1part2"}
    ]


def test_apt_install_delegates_to_sudo_install():
    """Shell 'apt install' commands route through sudo_install with the package name."""
    computer = SimpleNamespace(
        import_computer_api=False,
        import_skills=False,
        _has_imported_computer_api=False,
        _has_imported_skills=False,
        verbose=False,
        skills=SimpleNamespace(import_skills=mock.Mock()),
    )
    terminal = Terminal(computer=computer)
    with mock.patch.object(terminal, "sudo_install", return_value=True) as sudo_install:
        output = terminal.run("shell", "apt install cowsay", stream=False)
    sudo_install.assert_called_once_with("cowsay")
    assert "installed successfully" in output[0]["content"]


def test_streaming_run_parses_recipient_markers():
    """_streaming_run() splits @@@RECIPIENT/CONTENT markers into chunk fields."""
    computer = SimpleNamespace(
        import_computer_api=False,
        import_skills=False,
        _has_imported_computer_api=False,
        _has_imported_skills=False,
        verbose=False,
        skills=SimpleNamespace(import_skills=mock.Mock()),
    )
    terminal = Terminal(computer=computer)
    terminal._active_languages["fake"] = SimpleNamespace(
        run=lambda code: iter(
            [
                {
                    "type": "console",
                    "format": "output",
                    "content": "@@@RECIPIENT:user@@@CONTENT:hello@@@END",
                }
            ]
        )
    )
    chunks = list(terminal._streaming_run("fake", "x", display=False))
    assert chunks[0]["recipient"] == "user"
    assert chunks[0]["content"] == "hello"


def test_streaming_run_strips_hide_traceback_marker():
    """_streaming_run() removes traceback text before the @@@HIDE_TRACEBACK@@@ marker."""
    computer = SimpleNamespace(
        import_computer_api=False,
        import_skills=False,
        _has_imported_computer_api=False,
        _has_imported_skills=False,
        verbose=False,
        skills=SimpleNamespace(import_skills=mock.Mock()),
    )
    terminal = Terminal(computer=computer)
    terminal._active_languages["fake"] = SimpleNamespace(
        run=lambda code: iter(
            [
                {
                    "type": "console",
                    "format": "output",
                    "content": "Traceback...\n@@@HIDE_TRACEBACK@@@User-facing error",
                }
            ]
        )
    )
    chunks = list(terminal._streaming_run("fake", "x", display=False))
    assert "Traceback" not in chunks[0]["content"]
    assert "User-facing error" in chunks[0]["content"]


def _bare_computer(**overrides):
    """A computer stand-in with the flags Terminal.run() consults."""
    defaults = dict(
        import_computer_api=False,
        import_skills=False,
        _has_imported_computer_api=False,
        _has_imported_skills=False,
        verbose=False,
        skills=SimpleNamespace(import_skills=mock.Mock()),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_sudo_install_without_sudo_when_apt_succeeds():
    """sudo_install() returns True without prompting when plain apt works."""
    import subprocess

    terminal = Terminal(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.terminal.terminal.subprocess.run"
    ) as run:
        assert terminal.sudo_install("cowsay") is True
    run.assert_called_once_with(["apt", "install", "-y", "cowsay"], check=True)


def test_sudo_install_falls_back_to_sudo_password(capsys):
    """sudo_install() retries with sudo -S on CalledProcessError, feeding the password."""
    import subprocess

    terminal = Terminal(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.terminal.terminal.subprocess.run",
        side_effect=[
            subprocess.CalledProcessError(1, "apt"),
            mock.Mock(),
        ],
    ) as run, mock.patch(
        "interpreter.core.computer.terminal.terminal.getpass.getpass",
        return_value="pw",
    ):
        assert terminal.sudo_install("cowsay") is True
    assert run.call_args[0][0][:2] == ["sudo", "-S"]
    assert run.call_args[1]["input"] == b"pw"
    assert "Successfully installed cowsay" in capsys.readouterr().out


def test_sudo_install_returns_false_when_sudo_fails(capsys):
    """sudo_install() returns False and reports when both attempts fail."""
    import subprocess

    terminal = Terminal(computer=SimpleNamespace())
    with mock.patch(
        "interpreter.core.computer.terminal.terminal.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "apt"),
    ), mock.patch(
        "interpreter.core.computer.terminal.terminal.getpass.getpass",
        return_value="pw",
    ):
        assert terminal.sudo_install("cowsay") is False
    assert "Failed to install cowsay" in capsys.readouterr().out


def test_run_reports_failed_apt_install():
    """Terminal.run() reports failure when sudo_install() returns False."""
    terminal = Terminal(computer=_bare_computer())
    with mock.patch.object(terminal, "sudo_install", return_value=False):
        output = terminal.run("shell", "apt install cowsay", stream=False)
    assert "Failed to install package cowsay." in output[0]["content"]


def test_run_imports_computer_api_for_python_once(monkeypatch):
    """Python code mentioning computer triggers one computer-API import run."""
    run = mock.Mock()
    computer = _bare_computer(
        import_computer_api=True,
        run=run,
    )
    terminal = Terminal(computer=computer)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    # Terminal.run() skips the computer-API import when this variable is
    # "False" (the import payload itself sets it to stop recursive imports),
    # so a leaked value would skip the run() this test asserts on. Pin it
    # away to exercise the import path regardless of the environment.
    monkeypatch.delenv("INTERPRETER_COMPUTER_API", raising=False)

    def fake_streaming_run(language, code, display=False):
        yield {"type": "console", "format": "output", "content": "done"}

    with mock.patch.object(terminal, "_streaming_run", side_effect=fake_streaming_run):
        output = terminal.run("python", "computer.display.view()", stream=False)
    assert computer._has_imported_computer_api is True
    import_code = run.call_args[0][1] if run.call_args[0] else run.call_args[1]["code"]
    assert "INTERPRETER_COMPUTER_API" in import_code
    assert output[0]["content"] == "done"


def test_run_imports_skills_once_before_python():
    """Python runs import skills exactly once when enabled but not imported."""
    computer = _bare_computer(import_skills=True)
    terminal = Terminal(computer=computer)

    def fake_streaming_run(language, code, display=False):
        yield {"type": "console", "format": "output", "content": "done"}

    with mock.patch.object(terminal, "_streaming_run", side_effect=fake_streaming_run):
        terminal.run("python", "print(1)", stream=False)
        terminal.run("python", "print(2)", stream=False)
    assert computer._has_imported_skills is True
    computer.skills.import_skills.assert_called_once_with()


def test_run_stream_true_returns_the_generator():
    """Terminal.run(stream=True) hands back _streaming_run's generator unconsumed."""
    terminal = Terminal(computer=_bare_computer())
    with mock.patch.object(
        terminal, "_streaming_run", return_value=iter([{"a": 1}])
    ) as streaming:
        result = terminal.run("python", "print(1)", stream=True)
    streaming.assert_called_once_with("python", "print(1)", display=False)
    assert list(result) == [{"a": 1}]


def test_streaming_run_close_stops_languages():
    """Closing a _streaming_run generator routes GeneratorExit into stop()."""
    terminal = Terminal(computer=_bare_computer())

    def hanging_run(code):
        yield {"type": "console", "format": "output", "content": "x"}
        yield {"type": "console", "format": "output", "content": "y"}

    terminal._active_languages["fake"] = SimpleNamespace(run=hanging_run)
    stream = terminal._streaming_run("fake", "x", display=False)
    assert next(stream)["content"] == "x"
    with mock.patch.object(terminal, "stop") as stop:
        stream.close()
    stop.assert_called_once_with()


def test_terminate_skips_none_languages_and_clears():
    """terminate() tolerates None entries and drops every active language."""
    terminal = Terminal(computer=_bare_computer())
    language = mock.Mock()
    terminal._active_languages["real"] = language
    terminal._active_languages["ghost"] = None
    terminal.terminate()
    language.terminate.assert_called_once_with()
    assert terminal._active_languages == {}


@pytest.mark.parametrize(
    "code",
    [
        "apt install git curl",
        "apt install git && echo done",
        "apt install git\npip install requests",
        "apt-get install -y $PACKAGE",
        "apt install -y ${PKG}",
        "apt-get\ninstall git",
    ],
)
def test_apt_install_shortcut_ignores_compound_commands(code):
    """Only a bare one-package `apt install` uses the sudo_install shortcut."""
    terminal = Terminal(computer=_bare_computer())
    seen = []

    def fake_streaming_run(language, sent_code, display=False):
        seen.append((language, sent_code))
        return iter([])

    with mock.patch.object(terminal, "sudo_install") as sudo_install, \
            mock.patch.object(
                terminal, "_streaming_run", side_effect=fake_streaming_run
            ):
        terminal.run("bash", code, stream=False)

    sudo_install.assert_not_called()
    assert seen == [("shell", code)]


class _ChunkedShell:
    """Shell stub that yields one console-output chunk per item it is given."""

    name = "Shell"
    aliases = ["bash", "sh"]

    def __init__(self, computer):
        self.computer = computer

    def run(self, code):
        for piece in code.splitlines(keepends=True):
            yield {"type": "console", "format": "output", "content": piece}


def _chunked_terminal():
    terminal = Terminal(computer=_bare_computer())
    terminal.languages = [_ChunkedShell]
    return terminal


def test_shell_output_is_buffered_into_a_single_chunk():
    """Shell output arrives as one chunk, so the terminal is redrawn once (#304).

    Each line used to be printed separately, which made `cat`/`type` of a large
    file flicker through the whole file.
    """
    terminal = _chunked_terminal()

    chunks = list(terminal.run("shell", "a\nb\nc\n", stream=True))

    assert chunks == [
        {"type": "console", "format": "output", "content": "a\nb\nc\n"}
    ]


def test_non_shell_output_still_streams_per_chunk():
    """Buffering is scoped to shell: other languages keep streaming incrementally."""
    terminal = Terminal(computer=_bare_computer())
    terminal.languages = [type("Py", (_ChunkedShell,), {"name": "Python", "aliases": []})]

    chunks = list(terminal.run("python", "a\nb\n", stream=True))

    assert [chunk["content"] for chunk in chunks] == ["a\n", "b\n"]


def test_shell_output_flushes_once_the_buffer_is_full():
    """A long command still emits output mid-run instead of staying silent.

    The buffer is bounded so a `cat` of a huge file cannot exhaust memory and
    the user sees progress before the command finishes.
    """
    terminal = _chunked_terminal()
    piece = "x" * (_SHELL_OUTPUT_FLUSH_AT // 2)

    chunks = list(
        terminal.run("shell", f"{piece}\n{piece}\ntail\n", stream=True)
    )

    assert len(chunks) == 2
    assert chunks[0]["content"] == f"{piece}\n{piece}\n"
    assert chunks[1]["content"] == "tail\n"


class _IdleShell:
    """Shell stub that reports an idle queue between two output lines."""

    name = "Shell"
    aliases = ["bash", "sh"]
    emit_idle_ticks = False

    def __init__(self, computer):
        self.computer = computer

    def run(self, code):
        yield {"type": "console", "format": "output", "content": "started\n"}
        yield {"type": "console", "format": "output", "content": "", "_idle": True}
        yield {"type": "console", "format": "output", "content": "done\n"}


class _BrokenRecipientShell:
    """Shell stub whose second chunk makes parse_for_recipient raise."""

    name = "Shell"
    aliases = ["bash", "sh"]

    def __init__(self, computer):
        self.computer = computer

    def run(self, code):
        yield {"type": "console", "format": "output", "content": "saved\n"}
        yield {
            "type": "console",
            "format": "output",
            "content": "@@@RECIPIENT:user@@@END\n",
        }


def test_shell_output_flushes_when_the_producer_is_idle():
    """Buffered output is delivered while the command is still quiet.

    `echo started; sleep 600; echo done` used to keep "started" buffered for
    the whole sleep because only size or completion flushed it, so long
    commands lost their progress output. The backend now reports idleness and
    the terminal flushes on that, and the idle marker itself is consumed.
    """
    terminal = Terminal(computer=_bare_computer())
    terminal.languages = [_IdleShell]

    chunks = list(terminal.run("shell", "sleep 600", stream=True))

    assert [chunk["content"] for chunk in chunks] == ["started\n", "done\n"]
    assert all("_idle" not in chunk for chunk in chunks)
    assert terminal._active_languages["shell"].emit_idle_ticks is True


def test_shell_output_is_flushed_when_processing_raises():
    """Output already collected survives a later chunk failing to parse.

    The consumer never saw the buffer, so letting the exception propagate
    without flushing would discard valid output produced before the failure.
    """
    terminal = Terminal(computer=_bare_computer())
    terminal.languages = [_BrokenRecipientShell]

    delivered = []
    with pytest.raises(IndexError):
        for chunk in terminal.run("shell", "anything", stream=True):
            delivered.append(chunk)

    assert [chunk["content"] for chunk in delivered] == ["saved\n"]


class _ImageShell:
    """Shell stub that emits an image between two output lines."""

    name = "Shell"
    aliases = ["bash", "sh"]

    def __init__(self, computer):
        self.computer = computer

    def run(self, code):
        yield {"type": "console", "format": "output", "content": "before\n"}
        yield {"type": "image", "format": "base64", "content": "ZmFrZQ=="}
        yield {"type": "console", "format": "output", "content": "after\n"}


@pytest.mark.parametrize("stream", [True, False])
def test_shell_output_is_flushed_before_an_image_event(stream):
    """A non-text event is not yielded ahead of text buffered before it.

    The shell buffer only joins plain console output. An image chunk must
    therefore flush the pending text first, or the consumer (and the
    stream=False merge) sees the image before text that the command wrote
    earlier.
    """
    terminal = Terminal(computer=_bare_computer())
    terminal.languages = [_ImageShell]

    chunks = list(terminal.run("shell", "anything", stream=stream))

    assert [
        (chunk["type"], chunk.get("format"), chunk["content"]) for chunk in chunks
    ] == [
        ("console", "output", "before\n"),
        ("image", "base64", "ZmFrZQ=="),
        ("console", "output", "after\n"),
    ]


def test_recipient_tagged_shell_output_is_not_merged():
    """Output addressed to another recipient is emitted as its own chunk.

    Merging it would address the combined text to whichever recipient came
    first and lose the second tag.
    """
    terminal = _chunked_terminal()

    chunks = list(
        terminal.run(
            "shell",
            "@@@RECIPIENT:computer@@@CONTENT:to the computer@@@END\n"
            "plain\n"
            "@@@RECIPIENT:user@@@CONTENT:to the user@@@END\n",
            stream=True,
        )
    )

    assert [chunk.get("recipient") for chunk in chunks] == [
        "computer",
        None,
        "user",
    ]

