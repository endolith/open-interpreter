"""Smoke tests for the profile-loading pipeline.

``classic/develop`` ports a large set of profile defaults and renames profile
files; these tests pin the end-to-end load path (file -> dict -> applied to a
real ``OpenInterpreter``) so the port trips loudly if a profile stops loading.

The module computes its storage paths from ``oi_dir`` at import time, so each
test redirects ``profiles.profile_dir`` / ``profiles.oi_dir`` to a temp
directory and writes the profile files itself.
"""

import pytest
import requests
from unittest import mock

from interpreter.core.core import OpenInterpreter
from interpreter.terminal_interface.profiles import profiles


def _point_profiles_at(tmp_path, monkeypatch):
    """Redirect the profiles module's storage paths to a fresh temp dir."""
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(profiles, "oi_dir", str(tmp_path))
    monkeypatch.setattr(profiles, "profile_dir", str(profile_dir))
    return profile_dir


def test_profile_loads_yaml_and_applies_to_interpreter(tmp_path, monkeypatch):
    """profile() reads a local YAML file and applies it to a real interpreter."""
    profile_dir = _point_profiles_at(tmp_path, monkeypatch)
    (profile_dir / "myprofile.yaml").write_text(
        "version: 0.2.5\nllm:\n  temperature: 0.7\n"
    )

    interpreter = OpenInterpreter()
    result = profiles.profile(interpreter, "myprofile.yaml")

    assert result is interpreter
    assert interpreter.llm.temperature == 0.7


def test_profile_python_script_runs_with_bootstrap_stripped(tmp_path, monkeypatch):
    """A .py profile executes with the `interpreter` bootstrap lines removed."""
    profile_dir = _point_profiles_at(tmp_path, monkeypatch)
    (profile_dir / "script.py").write_text(
        "from interpreter import interpreter\n"
        "interpreter = OpenInterpreter()\n"
        "interpreter.verbose = True\n"
    )

    interpreter = OpenInterpreter()
    profiles.profile(interpreter, "script.py")

    assert interpreter.verbose is True


def test_profile_default_shorthand_loads_local_default(tmp_path, monkeypatch):
    """profile("default") resolves to the user's local default.yaml."""
    profile_dir = _point_profiles_at(tmp_path, monkeypatch)
    (profile_dir / "default.yaml").write_text("version: 0.2.5\noffline: True\n")

    interpreter = OpenInterpreter()
    profiles.profile(interpreter, "default")

    assert interpreter.offline is True


def test_profile_reserved_default_name_renames_custom_file(tmp_path, monkeypatch):
    """A user file at a reserved default profile name is renamed to
    {name}_custom.yaml so the packaged default is loaded instead."""
    profile_dir = _point_profiles_at(tmp_path, monkeypatch)
    (profile_dir / "fast.yaml").write_text("version: 0.2.5\noffline: True\n")

    interpreter = OpenInterpreter()
    profiles.profile(interpreter, "fast.yaml")

    assert not (profile_dir / "fast.yaml").exists()
    assert (profile_dir / "fast_custom.yaml").exists()
    # The packaged fast.yaml default is loaded, not the renamed custom file.
    assert interpreter.llm.model == "gpt-4o-mini"


def test_get_profile_loads_json(tmp_path, monkeypatch):
    """get_profile() parses .json profiles into a dict."""
    profile_dir = _point_profiles_at(tmp_path, monkeypatch)
    (profile_dir / "thing.json").write_text('{"llm": {"temperature": 0.2}}')

    loaded = profiles.get_profile("thing.json", str(profile_dir / "thing.json"))

    assert loaded == {"llm": {"temperature": 0.2}}


def test_write_key_to_profile_inserts_before_version_line(tmp_path, monkeypatch):
    """write_key_to_profile() adds a key just above the version trailer so the
    file stays a valid profile."""
    profile_dir = _point_profiles_at(tmp_path, monkeypatch)
    default_path = profile_dir / "default.yaml"
    default_path.write_text("offline: False\n\nversion: 0.2.5  # Profile version\n")
    monkeypatch.setattr(profiles, "user_default_profile_path", str(default_path))

    profiles.write_key_to_profile("auto_run", True)

    content = default_path.read_text()
    assert "auto_run: True" in content
    assert content.index("auto_run") < content.index("version: 0.2.5")
    assert content.endswith("version: 0.2.5  # Profile version\n")


def test_profile_missing_file_raises(tmp_path, monkeypatch):
    """profile() propagates the remote fetch error when a non-default profile is
    missing locally (the local file doesn't exist, so get_profile fetches it)."""
    _point_profiles_at(tmp_path, monkeypatch)

    response = mock.Mock()
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Client Error: Not Found for url: nonexistent.yaml"
    )
    monkeypatch.setattr(
        profiles.requests, "get", mock.Mock(return_value=response)
    )

    interpreter = OpenInterpreter()
    with pytest.raises(requests.exceptions.HTTPError, match="Not Found"):
        profiles.profile(interpreter, "nonexistent.yaml")
