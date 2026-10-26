from types import SimpleNamespace
from unittest import mock

from interpreter.core.terminal.terminal import Terminal


def test_get_language_by_name():
    """Terminal.get_language() resolves known language names and returns None for unknown ones."""
    terminal = Terminal(SimpleNamespace())
    assert terminal.get_language("python").name == "python"
    assert terminal.get_language("bash").name == "bash"
    assert terminal.get_language("unknown_xyz") is None


def test_get_language_is_case_insensitive():
    """Terminal.get_language() matches language names regardless of case."""
    terminal = Terminal(SimpleNamespace())
    assert terminal.get_language("BASH").name == "bash"


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
    terminal = Terminal(computer)

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
    terminal = Terminal(computer)
    with mock.patch.object(terminal, "sudo_install", return_value=True) as sudo_install:
        output = terminal.run("bash", "apt install cowsay", stream=False)
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
    terminal = Terminal(computer)
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
    terminal = Terminal(computer)
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
    """An interpreter stand-in with the toolbox flags Terminal.run() consults."""
    defaults = dict(
        import_toolbox_api=False,
        import_skills=False,
        _has_imported_toolbox_api=False,
        _has_imported_skills=False,
        verbose=False,
        skills=SimpleNamespace(import_skills=mock.Mock()),
    )
    defaults.update(overrides)
    return SimpleNamespace(toolbox=SimpleNamespace(**defaults))


def test_sudo_install_without_sudo_when_apt_succeeds():
    """sudo_install() returns True without prompting when plain apt works."""
    import subprocess

    terminal = Terminal(SimpleNamespace())
    with mock.patch(
        "interpreter.core.terminal.terminal.subprocess.run"
    ) as run:
        assert terminal.sudo_install("cowsay") is True
    run.assert_called_once_with(["apt", "install", "-y", "cowsay"], check=True)


def test_sudo_install_falls_back_to_sudo_password(capsys):
    """sudo_install() retries with sudo -S on CalledProcessError, feeding the password."""
    import subprocess

    terminal = Terminal(SimpleNamespace())
    with mock.patch(
        "interpreter.core.terminal.terminal.subprocess.run",
        side_effect=[
            subprocess.CalledProcessError(1, "apt"),
            mock.Mock(),
        ],
    ) as run, mock.patch(
        "interpreter.core.terminal.terminal.getpass.getpass",
        return_value="pw",
    ):
        assert terminal.sudo_install("cowsay") is True
    assert run.call_args[0][0][:2] == ["sudo", "-S"]
    assert run.call_args[1]["input"] == b"pw"
    assert "Successfully installed cowsay" in capsys.readouterr().out


def test_sudo_install_returns_false_when_sudo_fails(capsys):
    """sudo_install() returns False and reports when both attempts fail."""
    import subprocess

    terminal = Terminal(SimpleNamespace())
    with mock.patch(
        "interpreter.core.terminal.terminal.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "apt"),
    ), mock.patch(
        "interpreter.core.terminal.terminal.getpass.getpass",
        return_value="pw",
    ):
        assert terminal.sudo_install("cowsay") is False
    assert "Failed to install cowsay" in capsys.readouterr().out


def test_run_reports_failed_apt_install():
    """Terminal.run() reports failure when sudo_install() returns False."""
    terminal = Terminal(_bare_computer())
    with mock.patch.object(terminal, "sudo_install", return_value=False):
        output = terminal.run("bash", "apt install cowsay", stream=False)
    assert "Failed to install package cowsay." in output[0]["content"]


def test_run_imports_toolbox_api_for_python_once(monkeypatch):
    """Python code mentioning toolbox triggers one toolbox-API import run."""
    toolbox = SimpleNamespace(
        import_toolbox_api=True,
        _has_imported_toolbox_api=False,
        import_skills=False,
        _has_imported_skills=False,
        verbose=False,
        skills=SimpleNamespace(import_skills=mock.Mock()),
    )
    interpreter = SimpleNamespace(toolbox=toolbox)
    terminal = Terminal(interpreter)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    monkeypatch.delenv("INTERPRETER_TOOLBOX_API", raising=False)
    seen = []

    def fake_streaming_run(language, code, display=False):
        seen.append(code)
        if "__TOOLBOX_API_IMPORTED__" in code:
            yield {"type": "console", "format": "output", "content": "__TOOLBOX_API_IMPORTED__"}
        else:
            yield {"type": "console", "format": "output", "content": "done"}

    with mock.patch.object(terminal, "_streaming_run", side_effect=fake_streaming_run):
        output = terminal.run("python", "toolbox.display.view()", stream=False)
    assert toolbox._has_imported_toolbox_api is True
    assert any("INTERPRETER_TOOLBOX_API" in code for code in seen)
    assert output[0]["content"] == "done"


def test_run_imports_skills_once_before_python():
    """Python runs import skills exactly once when enabled but not imported."""
    toolbox = SimpleNamespace(
        import_toolbox_api=False,
        _has_imported_toolbox_api=False,
        import_skills=True,
        _has_imported_skills=False,
        verbose=False,
        skills=SimpleNamespace(import_skills=mock.Mock()),
    )
    interpreter = SimpleNamespace(toolbox=toolbox)
    terminal = Terminal(interpreter)

    def fake_streaming_run(language, code, display=False):
        yield {"type": "console", "format": "output", "content": "done"}

    with mock.patch.object(terminal, "_streaming_run", side_effect=fake_streaming_run):
        terminal.run("python", "print(1)", stream=False)
        terminal.run("python", "print(2)", stream=False)
    assert toolbox._has_imported_skills is True
    toolbox.skills.import_skills.assert_called_once_with()


def test_run_stream_true_returns_the_generator():
    """Terminal.run(stream=True) hands back _streaming_run's generator unconsumed."""
    terminal = Terminal(_bare_computer())
    with mock.patch.object(
        terminal, "_streaming_run", return_value=iter([{"a": 1}])
    ) as streaming:
        result = terminal.run("python", "print(1)", stream=True)
    streaming.assert_called_once_with("python", "print(1)", display=False)
    assert list(result) == [{"a": 1}]


def test_streaming_run_close_stops_languages():
    """Closing a _streaming_run generator routes GeneratorExit into stop()."""
    terminal = Terminal(_bare_computer())

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
    terminal = Terminal(_bare_computer())
    language = mock.Mock()
    terminal._active_languages["real"] = language
    terminal._active_languages["ghost"] = None
    terminal.terminate()
    language.terminate.assert_called_once_with()
    assert terminal._active_languages == {}
