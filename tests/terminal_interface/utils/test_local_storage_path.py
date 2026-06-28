from unittest import mock

from interpreter.terminal_interface.utils.local_storage_path import get_storage_path


def test_get_storage_path_default():
    with mock.patch(
        "interpreter.terminal_interface.utils.local_storage_path.config_dir",
        "/home/user/.config/open-interpreter",
    ):
        assert get_storage_path() == "/home/user/.config/open-interpreter"


def test_get_storage_path_subdirectory():
    with mock.patch(
        "interpreter.terminal_interface.utils.local_storage_path.config_dir",
        "/home/user/.config/open-interpreter",
    ):
        path = get_storage_path("profiles")
    assert path.endswith("profiles")
    assert path == "/home/user/.config/open-interpreter/profiles"
