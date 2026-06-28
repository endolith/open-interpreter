from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.terminal.terminal import Terminal


def test_get_language_by_name():
    terminal = Terminal(computer=SimpleNamespace())
    assert terminal.get_language("python").name == "Python"
    assert terminal.get_language("bash").name == "Shell"
    assert terminal.get_language("unknown_xyz") is None


def test_get_language_by_alias():
    terminal = Terminal(computer=SimpleNamespace())
    assert terminal.get_language("sh").name == "Shell"


def test_run_non_streaming_merges_output_chunks():
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
