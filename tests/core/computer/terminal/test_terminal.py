from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.terminal.terminal import Terminal


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
