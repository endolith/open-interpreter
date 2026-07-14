from interpreter.terminal_interface.utils.oi_dir import oi_dir


def test_oi_dir_is_open_interpreter_config_path():
    """oi_dir points at the open-interpreter user config directory."""
    assert isinstance(oi_dir, str)
    assert oi_dir
    assert "open-interpreter" in oi_dir
