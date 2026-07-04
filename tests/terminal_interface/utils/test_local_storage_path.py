import os
from unittest import mock

from interpreter.terminal_interface.utils.local_storage_path import get_storage_path

_CONFIG_DIR = "/home/user/.config/open-interpreter"


def test_get_storage_path_default():
    """get_storage_path with no argument returns the config directory."""
    with mock.patch(
        "interpreter.terminal_interface.utils.local_storage_path.config_dir",
        _CONFIG_DIR,
    ):
        assert get_storage_path() == _CONFIG_DIR


def test_get_storage_path_subdirectory():
    """get_storage_path joins the config directory with the requested subdirectory."""
    with mock.patch(
        "interpreter.terminal_interface.utils.local_storage_path.config_dir",
        _CONFIG_DIR,
    ):
        path = get_storage_path("profiles")
    assert path.endswith("profiles")
    assert path == os.path.join(_CONFIG_DIR, "profiles")
