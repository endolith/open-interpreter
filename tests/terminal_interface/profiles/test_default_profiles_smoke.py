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
