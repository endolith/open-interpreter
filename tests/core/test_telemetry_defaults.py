from interpreter import OpenInterpreter


def test_telemetry_disabled_by_default():
    """Open Interpreter should not send anonymous telemetry unless explicitly enabled."""
    interpreter = OpenInterpreter()
    assert interpreter.disable_telemetry is True
    assert interpreter.anonymous_telemetry is False


def test_telemetry_can_be_enabled():
    """Users can opt in to anonymous telemetry by setting disable_telemetry to False."""
    interpreter = OpenInterpreter(disable_telemetry=False)
    assert interpreter.disable_telemetry is False
    assert interpreter.anonymous_telemetry is True


def test_offline_mode_disables_telemetry_even_when_enabled():
    """Offline mode should disable telemetry even when disable_telemetry is False."""
    interpreter = OpenInterpreter(disable_telemetry=False, offline=True)
    assert interpreter.anonymous_telemetry is False
