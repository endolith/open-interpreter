"""Tests for the profile loading and application system.

Profiles configure the interpreter's LLM, computer, and behavior. These tests
cover profile resolution, loading from files and URLs, migration, and the
<<<<<<< ours
<<<<<<< ours
application of profile settings to interpreter objects.
=======
application of profile settings to interpreter objects. Tests document current
behavior only — no source changes.
>>>>>>> theirs
=======
application of profile settings to interpreter objects. Tests document current
behavior only — no source changes.
>>>>>>> theirs
"""

import json
from unittest import mock

import pytest

from interpreter.terminal_interface.profiles.profiles import (
    RemoveInterpreter,
    apply_profile,
    apply_profile_to_object,
    get_default_profile,
    get_profile,
    migrate_profile,
    profile,
    reset_profile,
)


def test_apply_profile_to_object_flat():
    """apply_profile_to_object sets flat attributes directly."""
    class Obj:
        def __init__(self):
            self.auto_run = False
            self.verbose = False

    obj = Obj()
    apply_profile_to_object(obj, {"auto_run": True, "verbose": True})
    assert obj.auto_run is True
    assert obj.verbose is True


def test_apply_profile_to_object_nested():
    """apply_profile_to_object recurses into nested dicts."""
    class Inner:
        def __init__(self):
            self.temperature = 0.0
            self.model = "gpt-4o"

    class Outer:
        def __init__(self):
            self.llm = Inner()

    obj = Outer()
    apply_profile_to_object(obj, {"llm": {"temperature": 0.7, "model": "gpt-4.1"}})
    assert obj.llm.temperature == 0.7
    assert obj.llm.model == "gpt-4.1"


def test_apply_profile_to_object_skips_wtf_dict():
    """apply_profile_to_object skips 'wtf' key when its value is a dict."""
    class Obj:
        def __init__(self):
            self.wtf = {"original": "value"}

    obj = Obj()
    apply_profile_to_object(obj, {"wtf": {"should": "not_apply"}})
    assert obj.wtf == {"original": "value"}


def test_remove_interpreter_strips_import():
    """RemoveInterpreter strips 'from interpreter import interpreter'."""
    import ast

    source = "from interpreter import interpreter\nx = 1\n"
    tree = ast.parse(source)
    transformed = RemoveInterpreter().visit(tree)
    ast.fix_missing_locations(transformed)
    result = ast.unparse(transformed)
    assert "from interpreter import interpreter" not in result
    assert "x = 1" in result


def test_remove_interpreter_strips_assignment():
    """RemoveInterpreter strips 'interpreter = OpenInterpreter()'."""
    import ast

    source = "interpreter = OpenInterpreter()\ny = 2\n"
    tree = ast.parse(source)
    transformed = RemoveInterpreter().visit(tree)
    ast.fix_missing_locations(transformed)
    result = ast.unparse(transformed)
    assert "OpenInterpreter()" not in result
    assert "y = 2" in result


def test_remove_interpreter_keeps_other_imports():
    """RemoveInterpreter keeps imports from other modules."""
    import ast

    source = "import os\nfrom interpreter import interpreter\n"
    tree = ast.parse(source)
    transformed = RemoveInterpreter().visit(tree)
    ast.fix_missing_locations(transformed)
    result = ast.unparse(transformed)
    assert "import os" in result
    assert "from interpreter import interpreter" not in result


def test_get_profile_local_yaml(tmp_path, monkeypatch):
    """get_profile loads a YAML profile from a local file."""
    profile_file = tmp_path / "test.yaml"
    profile_file.write_text("llm:\n  model: gpt-4.1\n")
    monkeypatch.setattr(
        "interpreter.terminal_interface.profiles.profiles.profile_dir",
        str(tmp_path),
    )
    result = get_profile("test.yaml", str(profile_file))
    assert result["llm"]["model"] == "gpt-4.1"


def test_get_profile_local_json(tmp_path, monkeypatch):
    """get_profile loads a JSON profile from a local file."""
    profile_file = tmp_path / "test.json"
    profile_file.write_text(json.dumps({"llm": {"model": "gpt-4.1"}}))
    monkeypatch.setattr(
        "interpreter.terminal_interface.profiles.profiles.profile_dir",
        str(tmp_path),
    )
    result = get_profile("test.json", str(profile_file))
    assert result["llm"]["model"] == "gpt-4.1"


def test_get_profile_local_python(tmp_path, monkeypatch):
    """get_profile loads a Python profile and strips interpreter bootstrap."""
    profile_file = tmp_path / "test.py"
    profile_file.write_text(
        "from interpreter import interpreter\n"
        "interpreter = OpenInterpreter()\n"
        "interpreter.auto_run = True\n"
        "interpreter.llm.model = 'gpt-4.1'\n"
    )
    monkeypatch.setattr(
        "interpreter.terminal_interface.profiles.profiles.profile_dir",
        str(tmp_path),
    )
    result = get_profile("test.py", str(profile_file))
    assert "start_script" in result
    assert "interpreter.auto_run = True" in result["start_script"]
    assert "from interpreter import interpreter" not in result["start_script"]
    assert "interpreter = OpenInterpreter()" not in result["start_script"]


def test_get_profile_url(monkeypatch):
    """get_profile fetches a profile from a URL."""
    mock_response = mock.MagicMock()
    mock_response.text = "llm:\n  model: gpt-4.1\n"
    mock_response.raise_for_status = mock.MagicMock()
    monkeypatch.setattr(
        "interpreter.terminal_interface.profiles.profiles.requests.get",
        lambda url: mock_response,
    )
    result = get_profile("https://example.com/profile.yaml", "/nonexistent")
    assert result["llm"]["model"] == "gpt-4.1"


def test_get_profile_not_found(monkeypatch):
    """get_profile raises when the profile doesn't exist locally or remotely."""
    import requests

    monkeypatch.setattr(
        "interpreter.terminal_interface.profiles.profiles.requests.get",
        lambda url: (_ for _ in ()).throw(requests.exceptions.HTTPError()),
    )
    with pytest.raises(Exception):
        get_profile("nonexistent.yaml", "/nonexistent/path.yaml")


def test_apply_profile_runs_start_script():
    """apply_profile executes the start_script in a scope with interpreter."""
    interpreter = mock.MagicMock()
    interpreter.computer.languages = []
    interpreter.auto_run = False
    profile_data = {"start_script": "interpreter.auto_run = True", "version": "0.2.5"}
    apply_profile(interpreter, profile_data, "/tmp/profile.yaml")
    assert interpreter.auto_run is True


@pytest.mark.xfail(reason="KNOWN BUG: apply_profile has `del profile[\"computer.languages\"]` instead of `del profile[\"computer\"][\"languages\"]`")
def test_apply_profile_filters_languages(monkeypatch):
    """apply_profile filters computer.languages to those listed in the profile."""
    interpreter = mock.MagicMock()

    class FakeLang:
        def __init__(self, name):
            self.name = name

    interpreter.computer.languages = [FakeLang("python"), FakeLang("shell"), FakeLang("javascript")]
    profile_data = {
        "computer": {"languages": ["python", "javascript"]},
        "llm": {"model": "gpt-4.1"},
        "version": "0.2.5",
    }
    apply_profile(interpreter, profile_data, "/tmp/profile.yaml")
    assert [lang.name for lang in interpreter.computer.languages] == ["python", "javascript"]
    assert interpreter.llm.model == "gpt-4.1"


def test_apply_profile_warns_about_system_message():
    """apply_profile displays a warning when system_message is in the profile."""
    interpreter = mock.MagicMock()
    interpreter.computer.languages = []
    interpreter.display_message = mock.MagicMock()
    profile_data = {"system_message": "Custom system message", "version": "0.2.5"}
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.time.sleep"
    ):
        apply_profile(interpreter, profile_data, "/tmp/profile.yaml")
    warning_calls = [c[0][0] for c in interpreter.display_message.call_args_list]
    assert any("system_message" in w for w in warning_calls)


def test_apply_profile_skips_migration_when_version_matches():
    """apply_profile skips migration prompt when version matches OI_VERSION."""
    interpreter = mock.MagicMock()
    interpreter.computer.languages = []
    profile_data = {"version": "0.2.5", "llm": {"model": "gpt-4.1"}}
    result = apply_profile(interpreter, profile_data, "/tmp/profile.yaml")
    assert result is interpreter


def test_profile_resolves_default_name():
    """profile() resolves shorthand names to default profile files."""
    interpreter = mock.MagicMock()
    interpreter.computer.languages = []
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.get_default_profile",
        return_value={"version": "0.2.5", "llm": {"model": "gpt-4.1"}},
    ):
        result = profile(interpreter, "local")
    assert result is interpreter


def test_profile_renames_reserved_name(tmp_path, monkeypatch):
    """profile() renames a user profile that conflicts with a default name."""
    monkeypatch.setattr(
        "interpreter.terminal_interface.profiles.profiles.profile_dir",
        str(tmp_path),
    )
    monkeypatch.setattr(
        "interpreter.terminal_interface.profiles.profiles.default_profiles_names",
        ["local.yaml"],
    )
    existing = tmp_path / "local.yaml"
    existing.write_text("old: true\n")
    interpreter = mock.MagicMock()
    interpreter.computer.languages = []
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.get_default_profile",
        return_value={"version": "0.2.5"},
    ):
        profile(interpreter, "local.yaml")
    assert (tmp_path / "local_custom.yaml").exists()


def test_migrate_profile_maps_old_keys():
    """migrate_profile reformats flat keys into nested dicts.

    KNOWN BUG: the source code builds reformatted_profile from the original
    profile instead of mapped_profile, so the attribute_mapping is computed
    but never used. This test documents the actual (buggy) behavior.
    """
    old_profile = {"model": "gpt-4o", "temperature": 0.5}
    mock_dump = mock.MagicMock()
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.yaml.safe_load",
        return_value=old_profile,
    ), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.yaml.dump",
        mock_dump,
    ), mock.patch("builtins.open", mock.mock_open()):
        migrate_profile("/old/path", "/new/path")
    assert mock_dump.called
    dumped_profile = mock_dump.call_args[0][0]
    assert dumped_profile == old_profile


def test_reset_profile_raises_for_unknown_profile():
    """reset_profile raises ValueError for unknown default profile names."""
    with pytest.raises(ValueError, match="not a default profile"):
        reset_profile("nonexistent.yaml")


def test_get_default_profile_yaml():
    """get_default_profile loads a YAML default profile."""
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.default_profiles_paths",
        ["/defaults/local.yaml"],
    ), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.oi_default_profiles_path",
        "/defaults",
    ), mock.patch(
        "builtins.open", mock.mock_open(read_data="llm:\n  model: gpt-4.1\n")
    ):
        result = get_default_profile("local.yaml")
    assert result["llm"]["model"] == "gpt-4.1"


def test_get_default_profile_python():
    """get_default_profile loads a Python default profile."""
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.default_profiles_paths",
        ["/defaults/custom.py"],
    ), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.oi_default_profiles_path",
        "/defaults",
    ), mock.patch(
        "builtins.open",
        mock.mock_open(read_data="interpreter.auto_run = True\n"),
    ):
        result = get_default_profile("custom.py")
    assert "start_script" in result
    assert "interpreter.auto_run = True" in result["start_script"]
