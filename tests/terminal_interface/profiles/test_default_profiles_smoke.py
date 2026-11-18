"""Smoke tests for every packaged default profile.

`profiles/defaults/` ships one config per model/provider (.py start-scripts
plus .yaml/.yml/.json configs). Loading a single representative file is already
covered elsewhere; what was missing is proof that *every* shipped file still
loads — a broken default would only surface when a user selects that profile.
These parametrized tests load each real file through get_default_profile and
check the returned shape and version.
"""

import ast
import os

import yaml

import pytest

from interpreter.terminal_interface.profiles import profiles


def _default_filenames():
    """Basenames of every file shipped in profiles/defaults/."""
    return sorted(os.path.basename(path) for path in profiles.default_profiles_paths)


@pytest.mark.parametrize("filename", _default_filenames())
def test_default_profile_loads(filename):
    """Each packaged default profile loads into a well-shaped dict.

    .py profiles come back as a start_script string plus the package version;
    .yaml/.yml/.json profiles come back as their parsed mapping. Any file that
    fails to parse means a profile users can select is broken on arrival.
    """
    result = profiles.get_default_profile(filename)

    assert isinstance(result, dict), filename
    extension = os.path.splitext(filename)[-1]
    if extension == ".py":
        assert isinstance(result["start_script"], str), filename
        assert result["version"] == profiles.OI_VERSION, filename
        ast.parse(result["start_script"])
    else:
        assert result, filename


@pytest.mark.parametrize("filename", _default_filenames())
def test_default_profile_reports_current_version(filename):
    """Every packaged default profile loads as the current version.

    apply_profile reads `version` to decide whether a profile is a stale user
    file and, if it is, blocks on a migration prompt before applying anything.
    Profiles shipped inside the package are current by construction, so a
    packaged profile must never load without the current version — otherwise
    selecting it prompts to migrate a file the user does not own.
    """
    assert profiles.get_default_profile(filename)["version"] == profiles.OI_VERSION


def _yaml_default_names():
    """Basenames of the packaged defaults written as YAML."""
    return [name for name in _default_filenames() if name.endswith((".yaml", ".yml"))]


def _default_path(filename):
    """Absolute path of a packaged default profile."""
    return next(
        path for path in profiles.default_profiles_paths if os.path.basename(path) == filename
    )


@pytest.mark.parametrize("filename", _yaml_default_names())
def test_default_yaml_profiles_place_max_output_at_the_top_level(filename):
    """A commented max_output instruction must sit at the top level, not under llm.

    apply_profile_to_object only assigns attributes that appear at the profile's
    top level, so a max_output left inside the `llm:` block silently did nothing
    when a user followed the shipped instruction and uncommented it. Three
    packaged templates had it nested under llm; uncommenting the line here must
    produce a profile the interpreter actually reads.
    """
    with open(_default_path(filename), encoding="utf-8") as handle:
        text = handle.read()

    commented = [line for line in text.splitlines() if line.lstrip().startswith("# max_output:")]
    assert [line for line in commented if line != line.lstrip()] == [], filename
    if not commented:
        return

    assert len(commented) == 1, filename
    profile = yaml.safe_load(text.replace(commented[0], commented[0].lstrip("# ")))
    assert profile["max_output"] == 2500, filename
