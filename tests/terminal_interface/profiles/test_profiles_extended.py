"""Extended unit tests for the profiles module.

``classic/develop`` rewrites profiles.py (+133 lines) and renames profile
defaults, so these tests pin the pure-logic paths that the loading smoke
tests do not reach: migration, reset, version detection, URL shortcut
resolution, and the AST bootstrap stripper. All paths are exercised without
real kernels, LLMs, or network access (requests is mocked).
"""

import ast
import os
from unittest import mock

import pytest
import requests

from interpreter.core.core import OpenInterpreter
from interpreter.terminal_interface.profiles import profiles


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    """Redirect the profiles module's storage paths to a temp dir and return
    the paths plus the defaults dir."""
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir(exist_ok=True)
    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(profiles, "oi_dir", str(tmp_path))
    monkeypatch.setattr(profiles, "profile_dir", str(profile_dir))
    monkeypatch.setattr(profiles, "oi_default_profiles_path", str(defaults_dir))
    return {"profile_dir": str(profile_dir), "defaults_dir": str(defaults_dir)}


def _set_defaults(defaults_dir, files, monkeypatch):
    """Write default profile files into the defaults dir and point the module at them."""
    for name, content in files.items():
        with open(os.path.join(defaults_dir, name), "w") as f:
            f.write(content)
    monkeypatch.setattr(
        profiles, "default_profiles_paths",
        [os.path.join(defaults_dir, name) for name in files],
    )
    monkeypatch.setattr(profiles, "default_profiles_names", list(files))


# ---------------------------------------------------------------------------
# get_profile: URL shortcuts and remote fetch
# ---------------------------------------------------------------------------


def test_get_profile_i_com_shortcut_probes_extensions(profile_env):
    """i.com/ URLs are rewritten to openinterpreter.com and the extension is
    probed (.json, then .py) until a non-404 response is found."""
    tried = []

    def fake_get(url, **kwargs):
        tried.append(url)
        if url.endswith(".json"):
            response = mock.Mock()
            response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "404"
            )
            return response
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.text = "interpreter.verbose = True"
        return response

    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.requests.get",
        side_effect=fake_get,
    ):
        result = profiles.get_profile(
            "i.com/foo", os.path.join(profile_env["profile_dir"], "i.com/foo")
        )

    assert tried[0] == "https://openinterpreter.com/profiles/foo.json"
    assert tried[1] == "https://openinterpreter.com/profiles/foo.py"
    assert result == {"start_script": "interpreter.verbose = True", "version": "0.2.5"}


def test_get_profile_fetches_python_url(profile_env):
    """A remote .py URL profile is fetched and returned as a start_script."""
    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.text = "interpreter.offline = True"
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.requests.get",
        return_value=response,
    ):
        result = profiles.get_profile(
            "https://example.com/p.py", os.path.join(profile_env["profile_dir"], "p.py")
        )
    assert result == {"start_script": "interpreter.offline = True", "version": "0.2.5"}


def test_get_profile_fetches_yaml_url(profile_env):
    """A remote .yaml URL profile is parsed with yaml.safe_load."""
    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.text = "offline: True"
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.requests.get",
        return_value=response,
    ):
        result = profiles.get_profile(
            "https://example.com/p.yaml",
            os.path.join(profile_env["profile_dir"], "p.yaml"),
        )
    assert result == {"offline": True}


def test_get_profile_unknown_extension_raises(profile_env):
    """A remote profile with an unsupported extension raises a descriptive error."""
    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.text = "stuff"
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.requests.get",
        return_value=response,
    ):
        with pytest.raises(Exception, match="not found"):
            profiles.get_profile(
                "https://example.com/p.xyz",
                os.path.join(profile_env["profile_dir"], "p.xyz"),
            )


def test_get_profile_fetches_json_url(profile_env):
    """A remote .json URL profile is parsed with json.loads."""
    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.text = '{"offline": true}'
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.requests.get",
        return_value=response,
    ):
        result = profiles.get_profile(
            "https://example.com/p.json",
            os.path.join(profile_env["profile_dir"], "p.json"),
        )
    assert result == {"offline": True}


def test_profile_default_resets_and_retries_when_missing(profile_env, monkeypatch):
    """profile('default') resets the default profile and retries when the local
    file is missing (get_profile fails once)."""
    _set_defaults(
        profile_env["defaults_dir"], {"default.yaml": "offline: True\n"}, monkeypatch
    )
    calls = {"n": 0}

    def flaky(name, path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("boom")
        return {"version": "0.2.5", "offline": True}

    interpreter = mock.Mock()
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.get_profile",
        side_effect=flaky,
    ):
        profiles.profile(interpreter, "default")
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# RemoveInterpreter AST stripper
# ---------------------------------------------------------------------------


def test_remove_interpreter_strips_import_from():
    """RemoveInterpreter removes `from interpreter import interpreter`."""
    tree = profiles.RemoveInterpreter().visit(
        ast.parse("from interpreter import interpreter\nimport os")
    )
    assert [type(n) for n in tree.body] == [ast.Import]


def test_remove_interpreter_strips_bootstrap_assign():
    """RemoveInterpreter removes `interpreter = OpenInterpreter()` but keeps
    other assignments."""
    tree = profiles.RemoveInterpreter().visit(
        ast.parse("interpreter = OpenInterpreter()\nx = 1")
    )
    assert ast.unparse(tree) == "x = 1"


def test_remove_interpreter_keeps_unrelated_import_from():
    """RemoveInterpreter keeps `from interpreter import something_else` and
    imports from other modules."""
    tree = profiles.RemoveInterpreter().visit(
        ast.parse("from interpreter import something_else\nfrom os import path")
    )
    assert len(tree.body) == 2


# ---------------------------------------------------------------------------
# apply_profile: migration prompt, version rewrite, system message FYI
# ---------------------------------------------------------------------------


def test_apply_profile_skips_migration_on_no(profile_env):
    """apply_profile() with a version mismatch and a 'n' answer skips loading
    and appends the version trailer."""
    path = os.path.join(profile_env["profile_dir"], "default.yaml")
    with open(path, "w") as f:
        f.write("version: 0.2.0\n")
    interpreter = mock.Mock()
    with mock.patch("builtins.input", return_value="n"):
        result = profiles.apply_profile(
            interpreter, {"version": "0.2.0", "llm": {"model": "x"}}, path
        )
    assert result is interpreter
    assert "version: 0.2.5" in open(path).read()


def test_apply_profile_migrates_and_rewrites_gpt4(profile_env):
    """apply_profile() with 'y' migrates the app directory and rewrites a
    gpt-4 default.yaml to gpt-4.1."""
    path = os.path.join(profile_env["profile_dir"], "default.yaml")
    with open(path, "w") as f:
        f.write("version: 0.2.0\nllm:\n  model: gpt-4\n")
    interpreter = mock.Mock()
    with mock.patch("builtins.input", return_value="y"), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.migrate_user_app_directory"
    ) as migrate:
        result = profiles.apply_profile(
            interpreter, {"version": "0.2.0", "llm": {"model": "gpt-4"}}, path
        )
    assert migrate.called
    assert result.llm.model == "gpt-4.1"
    assert "gpt-4.1" in open(path).read()


def test_apply_profile_skips_default_yaml_rewrite_for_other_file(profile_env):
    """apply_profile() migrates but does not rewrite the file for non-default
    profile paths."""
    path = os.path.join(profile_env["profile_dir"], "other.yaml")
    with open(path, "w") as f:
        f.write("version: 0.2.0\n")
    interpreter = mock.Mock()
    with mock.patch("builtins.input", return_value="y"), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.migrate_user_app_directory"
    ):
        profiles.apply_profile(
            interpreter, {"version": "0.2.0", "llm": {"model": "gpt-4"}}, path
        )
    # The non-default file keeps its old version text.
    assert "version: 0.2.0" in open(path).read()


def test_apply_profile_rewrites_gpt4_turbo_preview(profile_env):
    """apply_profile() rewrites gpt-4-turbo-preview to gpt-4.1 in default.yaml."""
    path = os.path.join(profile_env["profile_dir"], "default.yaml")
    with open(path, "w") as f:
        f.write("version: 0.2.0\nllm:\n  model: gpt-4-turbo-preview\n")
    interpreter = mock.Mock()
    with mock.patch("builtins.input", return_value="y"), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.migrate_user_app_directory"
    ):
        result = profiles.apply_profile(
            interpreter,
            {"version": "0.2.0", "llm": {"model": "gpt-4-turbo-preview"}},
            path,
        )
    assert result.llm.model == "gpt-4.1"
    assert "gpt-4.1" in open(path).read()


def test_apply_profile_warns_on_system_message(profile_env):
    """apply_profile() displays an FYI when the profile overrides system_message."""
    path = os.path.join(profile_env["profile_dir"], "p.yaml")
    interpreter = mock.Mock()
    with mock.patch("interpreter.terminal_interface.profiles.profiles.time.sleep"):
        profiles.apply_profile(
            interpreter, {"version": "0.2.5", "system_message": "custom"}, path
        )
    assert interpreter.display_message.call_count >= 2


def test_apply_profile_filters_computer_languages(profile_env):
    """apply_profile() filters computer.languages to the requested names and
    drops the languages key without error (regression test for #225)."""
    path = os.path.join(profile_env["profile_dir"], "p.yaml")
    interpreter = mock.Mock()
    interpreter.computer.languages = [
        mock.Mock(name="python"),
        mock.Mock(name="javascript"),
    ]
    interpreter.computer.languages[0].name = "python"
    interpreter.computer.languages[1].name = "javascript"
    with mock.patch("interpreter.terminal_interface.profiles.profiles.time.sleep"):
        profiles.apply_profile(
            interpreter,
            {"version": "0.2.5", "computer": {"languages": ["javascript"]}},
            path,
        )
    assert [l.name for l in interpreter.computer.languages] == ["javascript"]


def test_apply_profile_raises_when_llm_not_dict(profile_env):
    """apply_profile() propagates the error when profile['llm'] is not a dict
    during the model-rewrite (the bare except/raise preserves the original)."""
    path = os.path.join(profile_env["profile_dir"], "default.yaml")
    with open(path, "w") as f:
        f.write("version: 0.2.0\n")
    interpreter = mock.Mock()
    with mock.patch("builtins.input", return_value="y"), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.migrate_user_app_directory"
    ):
        with pytest.raises(TypeError, match="string indices"):
            profiles.apply_profile(
                interpreter, {"version": "0.2.0", "llm": "not-a-dict"}, path
            )


# ---------------------------------------------------------------------------
# apply_profile_to_object
# ---------------------------------------------------------------------------


def test_apply_profile_to_object_sets_nested_and_flat():
    """apply_profile_to_object() recurses into nested dicts and sets flat keys."""
    interpreter = mock.Mock()
    interpreter.llm = mock.Mock()
    profiles.apply_profile_to_object(
        interpreter, {"llm": {"temperature": 0.7}, "auto_run": False}
    )
    assert interpreter.llm.temperature == 0.7
    assert interpreter.auto_run is False


def test_apply_profile_to_object_skips_wtf_key():
    """apply_profile_to_object() skips the special 'wtf' key."""
    interpreter = mock.Mock(spec=[])
    profiles.apply_profile_to_object(interpreter, {"wtf": {"x": 1}})
    assert not hasattr(interpreter, "wtf")


# ---------------------------------------------------------------------------
# open_storage_dir
# ---------------------------------------------------------------------------


def test_open_storage_dir_uses_xdg_open_on_linux():
    """open_storage_dir() uses xdg-open on Linux."""
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.platform.system",
        return_value="Linux",
    ), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.subprocess.call"
    ) as subprocess_call:
        profiles.open_storage_dir("profiles")
    assert subprocess_call.call_args[0][0][0] == "xdg-open"


def test_open_storage_dir_falls_back_to_open():
    """open_storage_dir() falls back to `open` when xdg-open is missing."""
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.platform.system",
        return_value="Linux",
    ), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.subprocess.call",
        side_effect=[FileNotFoundError, None],
    ) as subprocess_call:
        profiles.open_storage_dir("profiles")
    assert subprocess_call.call_args_list[1][0][0][0] == "open"


def test_open_storage_dir_uses_startfile_on_windows():
    """open_storage_dir() uses os.startfile on Windows."""
    fake_os = mock.MagicMock()
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.platform.system",
        return_value="Windows",
    ), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.os", fake_os
    ):
        profiles.open_storage_dir("profiles")
    assert fake_os.startfile.called


# ---------------------------------------------------------------------------
# reset_profile
# ---------------------------------------------------------------------------


def test_reset_profile_creates_default_file(profile_env, monkeypatch):
    """reset_profile() copies the packaged default into an empty profiles dir."""
    _set_defaults(profile_env["defaults_dir"], {"default.yaml": "offline: True\n"}, monkeypatch)
    profiles.reset_profile("default.yaml")
    target = os.path.join(profile_env["profile_dir"], "default.yaml")
    assert os.path.exists(target)
    assert "offline: True" in open(target).read()


def test_reset_profile_rejects_unknown_name(profile_env, monkeypatch):
    """reset_profile() raises for a name that is not a packaged default."""
    _set_defaults(profile_env["defaults_dir"], {"default.yaml": "offline: True\n"}, monkeypatch)
    with pytest.raises(ValueError, match="not a default profile"):
        profiles.reset_profile("nope.yaml")


def test_reset_profile_skips_non_default_yaml(profile_env, monkeypatch):
    """reset_profile() only resets default.yaml, ignoring other defaults."""
    _set_defaults(
        profile_env["defaults_dir"],
        {"default.yaml": "offline: True\n", "fast.yaml": "model: gpt\n"},
        monkeypatch,
    )
    profiles.reset_profile("fast.yaml")
    # Only default.yaml is copied; fast.yaml is a python-package profile.
    assert not os.path.exists(
        os.path.join(profile_env["profile_dir"], "fast.yaml")
    )


def test_reset_profile_interactive_no(profile_env, monkeypatch):
    """reset_profile() does not overwrite when the user answers 'n'."""
    _set_defaults(profile_env["defaults_dir"], {"default.yaml": "offline: True\n"}, monkeypatch)
    target = os.path.join(profile_env["profile_dir"], "default.yaml")
    with open(target, "w") as f:
        f.write("auto_run: True\n")
    with mock.patch("builtins.input", return_value="n"), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.determine_user_version",
        return_value="0.2.5",
    ):
        profiles.reset_profile("default.yaml")
    assert open(target).read().strip() == "auto_run: True"


def test_reset_profile_interactive_yes_trashes_and_copies(profile_env, monkeypatch):
    """reset_profile() moves the old file to trash and copies the default on 'y'."""
    _set_defaults(profile_env["defaults_dir"], {"default.yaml": "offline: True\n"}, monkeypatch)
    target = os.path.join(profile_env["profile_dir"], "default.yaml")
    with open(target, "w") as f:
        f.write("auto_run: True\n")
    trashed = []
    with mock.patch("builtins.input", return_value="y"), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.determine_user_version",
        return_value="0.2.5",
    ), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.send2trash.send2trash",
        side_effect=lambda p: trashed.append(p),
    ):
        profiles.reset_profile("default.yaml")
    assert trashed and os.path.basename(trashed[0]) == "default.yaml"
    assert open(target).read().strip() == "offline: True"


def test_reset_profile_creates_directories_when_missing(tmp_path, monkeypatch):
    """reset_profile() makes the profiles dir (and oi_dir) when they are absent."""
    oi_dir = tmp_path / "oi"
    profile_dir = oi_dir / "profiles"
    monkeypatch.setattr(profiles, "oi_dir", str(oi_dir))
    monkeypatch.setattr(profiles, "profile_dir", str(profile_dir))
    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir()
    _set_defaults(
        str(defaults_dir), {"default.yaml": "offline: True\n"}, monkeypatch
    )
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.determine_user_version",
        return_value=None,
    ):
        profiles.reset_profile("default.yaml")
    assert profile_dir.exists()
    assert (profile_dir / "default.yaml").exists()


def test_reset_profile_resets_known_historical_profile(tmp_path, monkeypatch):
    """reset_profile() copies the default without prompting when the current
    file is a known historical profile."""
    oi_dir = tmp_path / "oi"
    profile_dir = oi_dir / "profiles"
    profile_dir.mkdir(parents=True)
    monkeypatch.setattr(profiles, "oi_dir", str(oi_dir))
    monkeypatch.setattr(profiles, "profile_dir", str(profile_dir))
    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir()
    _set_defaults(
        str(defaults_dir), {"default.yaml": "offline: True\n"}, monkeypatch
    )
    historical = "old known profile content"
    monkeypatch.setattr(profiles, "historical_profiles", [historical])
    target = profile_dir / "default.yaml"
    target.write_text(historical)
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.determine_user_version",
        return_value="0.2.5",
    ):
        profiles.reset_profile("default.yaml")
    assert target.read_text().strip() == "offline: True"


# ---------------------------------------------------------------------------
# get_default_profile
# ---------------------------------------------------------------------------


def test_get_default_profile_yaml(profile_env, monkeypatch):
    """get_default_profile() parses the packaged .yaml default."""
    _set_defaults(profile_env["defaults_dir"], {"fast.yaml": "model: gpt\n"}, monkeypatch)
    assert profiles.get_default_profile("fast.yaml") == {"model": "gpt"}


def test_get_default_profile_python(profile_env, monkeypatch):
    """get_default_profile() strips bootstrap from a packaged .py default."""
    _set_defaults(
        profile_env["defaults_dir"],
        {
            "script.py": "from interpreter import interpreter\n"
            "interpreter = OpenInterpreter()\n"
            "interpreter.verbose = True\n"
        },
        monkeypatch,
    )
    result = profiles.get_default_profile("script.py")
    assert result["version"] == "0.2.5"
    assert "verbose" in result["start_script"]
    assert "interpreter = OpenInterpreter()" not in result["start_script"]


def test_get_default_profile_json(profile_env, monkeypatch):
    """get_default_profile() parses a packaged .json default."""
    _set_defaults(
        profile_env["defaults_dir"], {"fast.json": '{"model": "gpt"}\n'}, monkeypatch
    )
    assert profiles.get_default_profile("fast.json") == {"model": "gpt"}


# ---------------------------------------------------------------------------
# determine_user_version
# ---------------------------------------------------------------------------


def test_determine_user_version_reads_profile(profile_env):
    """determine_user_version() returns the version key from default.yaml."""
    os.makedirs(os.path.join(profile_env["profile_dir"]), exist_ok=True)
    with open(
        os.path.join(profile_env["profile_dir"], "default.yaml"), "w"
    ) as f:
        f.write("version: 0.2.5\n")
    assert profiles.determine_user_version() == "0.2.5"


def test_determine_user_version_old_dirs(tmp_path, monkeypatch):
    """determine_user_version() detects legacy config directories."""
    monkeypatch.setattr(profiles, "oi_dir", str(tmp_path / "oi"))
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.platformdirs.user_config_dir",
        side_effect=lambda name: os.path.join(str(tmp_path), name),
    ):
        os.makedirs(os.path.join(str(tmp_path), "Open Interpreter"))
        assert profiles.determine_user_version() == "pre_0.2.0"

    monkeypatch.setattr(profiles, "oi_dir", str(tmp_path / "oi"))
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.platformdirs.user_config_dir",
        side_effect=lambda name: os.path.join(str(tmp_path), name),
    ):
        os.makedirs(os.path.join(str(tmp_path), "Open Interpreter Terminal"))
        assert profiles.determine_user_version() == "0.2.0"


def test_determine_user_version_none(tmp_path, monkeypatch):
    """determine_user_version() returns None when no directory exists."""
    monkeypatch.setattr(profiles, "oi_dir", str(tmp_path / "oi"))
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.platformdirs.user_config_dir",
        side_effect=lambda name: os.path.join(str(tmp_path), name),
    ):
        assert profiles.determine_user_version() is None


# ---------------------------------------------------------------------------
# migrate_profile
# ---------------------------------------------------------------------------


def test_migrate_profile_writes_comment_wrapper(profile_env):
    """migrate_profile() writes a comment-wrapped profile with the version trailer."""
    old = os.path.join(profile_env["profile_dir"], "old.yaml")
    new = os.path.join(profile_env["profile_dir"], "new.yaml")
    with open(old, "w") as f:
        f.write("model: gpt-4\n")
    profiles.migrate_profile(old, new)
    content = open(new).read()
    assert "### OPEN INTERPRETER PROFILE" in content
    assert "version: 0.2.5" in content
    assert "model: gpt-4" in content


def test_migrate_profile_reformats_dotted_keys(profile_env):
    """migrate_profile() nests dotted keys into a dict structure."""
    old = os.path.join(profile_env["profile_dir"], "old.yaml")
    new = os.path.join(profile_env["profile_dir"], "new.yaml")
    with open(old, "w") as f:
        f.write("llm.temperature: 0.7\nllm.model: gpt-4\n")
    profiles.migrate_profile(old, new)
    content = open(new).read()
    assert "temperature: 0.7" in content


def _old_system_messages():
    """Extract the old_system_messages list from migrate_profile's source so the
    migration tests exercise the real literals instead of duplicating them."""
    import ast
    import inspect

    src = inspect.getsource(profiles.migrate_profile)
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "old_system_messages"
        ):
            return [ast.literal_eval(element) for element in node.value.elts]
    raise AssertionError("old_system_messages not found in migrate_profile")


def _write_yaml_multiline(path, key, value):
    """Write a YAML file with `key: |` and the value as an indented block."""
    indented = "\n".join("  " + line for line in value.split("\n"))
    with open(path, "w") as f:
        f.write(f"{key}: |\n{indented}\n")


def test_migrate_profile_drops_exact_old_system_message(profile_env):
    """migrate_profile() deletes a system_message that exactly matches a known
    old default (after normalization)."""
    old = os.path.join(profile_env["profile_dir"], "old.yaml")
    new = os.path.join(profile_env["profile_dir"], "new.yaml")
    _write_yaml_multiline(old, "system_message", _old_system_messages()[9])
    profiles.migrate_profile(old, new)
    # The old system message must be gone from the profile body.
    body = open(new).read().split("# Be sure to remove")[0]
    assert "system_message" not in body


def test_migrate_profile_extracts_custom_instructions(profile_env):
    """migrate_profile() turns the tail of a prefixed old system message into
    custom_instructions."""
    old = os.path.join(profile_env["profile_dir"], "old.yaml")
    new = os.path.join(profile_env["profile_dir"], "new.yaml")
    _write_yaml_multiline(old, "system_message", _old_system_messages()[9] + " Always be brief.")
    profiles.migrate_profile(old, new)
    body = open(new).read().split("# Be sure to remove")[0]
    assert "custom_instructions" in body
    assert "Always be brief." in body
    assert "system_message" not in body


def test_migrate_profile_keeps_custom_system_message(profile_env):
    """migrate_profile() leaves a custom system_message untouched when it is not
    a known old default."""
    old = os.path.join(profile_env["profile_dir"], "old.yaml")
    new = os.path.join(profile_env["profile_dir"], "new.yaml")
    with open(old, "w") as f:
        f.write("system_message: 'totally custom'\n")
    profiles.migrate_profile(old, new)
    body = open(new).read().split("# Be sure to remove")[0]
    assert "system_message" in body
    assert "totally custom" in body


# ---------------------------------------------------------------------------
# migrate_app_directory / migrate_user_app_directory
# ---------------------------------------------------------------------------


def test_migrate_app_directory_copies_and_migrates(profile_env):
    """migrate_app_directory() copies profiles, conversations, and config.yaml."""
    old_dir = os.path.join(profile_env["profile_dir"], "..", "old")
    os.makedirs(os.path.join(old_dir, "profiles"))
    os.makedirs(os.path.join(old_dir, "conversations"))
    with open(os.path.join(old_dir, "profiles", "custom.yaml"), "w") as f:
        f.write("model: gpt-4\n")
    with open(os.path.join(old_dir, "profiles", "notes.txt"), "w") as f:
        f.write("plain text\n")
    with open(os.path.join(old_dir, "conversations", "conv.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(old_dir, "config.yaml"), "w") as f:
        f.write("model: gpt-4\n")
    new_dir = os.path.join(profile_env["profile_dir"], "new")
    os.makedirs(new_dir)
    profiles.migrate_app_directory(old_dir, new_dir, os.path.join(new_dir, "profiles"))
    new_profiles = os.path.join(new_dir, "profiles")
    assert "custom.yaml" in os.listdir(new_profiles)
    assert "default.yaml" in os.listdir(new_profiles)
    assert "notes.txt" in os.listdir(new_profiles)
    assert os.path.exists(os.path.join(new_dir, "conversations", "conv.json"))


def test_migrate_app_directory_appends_missing_version(tmp_path):
    """migrate_app_directory() appends a version trailer to migrated yaml files
    that have none."""
    old_dir = tmp_path / "old"
    os.makedirs(old_dir / "profiles")
    with open(old_dir / "profiles" / "nover.yaml", "w") as f:
        f.write("model: gpt-4\n")
    new_dir = tmp_path / "new"
    os.makedirs(new_dir)
    profiles.migrate_app_directory(
        str(old_dir), str(new_dir), str(new_dir / "profiles")
    )
    migrated = (new_dir / "profiles" / "nover.yaml").read_text()
    # migrate_profile writes the comment wrapper which always includes a version.
    assert "version: 0.2.5" in migrated


def test_migrate_app_directory_appends_version_when_migration_omits_it(tmp_path, monkeypatch):
    """The post-migration guard appends a version trailer to any yaml that
    still lacks one (e.g. a profile that bypassed migrate_profile)."""
    old_dir = tmp_path / "old"
    os.makedirs(old_dir / "profiles")
    with open(old_dir / "profiles" / "bare.yaml", "w") as f:
        f.write("model: gpt-4\n")

    # Simulate a migration that copied the file verbatim without adding a version.
    monkeypatch.setattr(
        profiles,
        "migrate_profile",
        lambda old_path, new_path: open(new_path, "w").write(open(old_path).read()),
    )
    new_dir = tmp_path / "new"
    os.makedirs(new_dir)
    profiles.migrate_app_directory(
        str(old_dir), str(new_dir), str(new_dir / "profiles")
    )
    migrated = (new_dir / "profiles" / "bare.yaml").read_text()
    assert "version: 0.2.1" in migrated


def test_migrate_user_app_directory_pre_020(profile_env):
    """migrate_user_app_directory() migrates from the pre-0.2.0 directory."""
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.determine_user_version",
        return_value="pre_0.2.0",
    ), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.migrate_app_directory"
    ) as migrate:
        profiles.migrate_user_app_directory()
    assert migrate.call_args[0][0].endswith("Open Interpreter")


def test_migrate_user_app_directory_020(profile_env):
    """migrate_user_app_directory() migrates from the 0.2.0 directory."""
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.determine_user_version",
        return_value="0.2.0",
    ), mock.patch(
        "interpreter.terminal_interface.profiles.profiles.migrate_app_directory"
    ) as migrate:
        profiles.migrate_user_app_directory()
    assert migrate.call_args[0][0].endswith("Open Interpreter Terminal")


# ---------------------------------------------------------------------------
# write_key_to_profile
# ---------------------------------------------------------------------------


def test_write_key_to_profile_fails_silently_on_missing_file(profile_env):
    """write_key_to_profile() swallows errors when the default profile is missing."""
    with mock.patch(
        "interpreter.terminal_interface.profiles.profiles.user_default_profile_path",
        os.path.join(profile_env["profile_dir"], "missing.yaml"),
    ):
        profiles.write_key_to_profile("auto_run", True)  # should not raise
