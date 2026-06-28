from interpreter.core.computer.terminal.languages.applescript import AppleScript


def test_applescript_add_active_line_indicators():
    script = AppleScript()
    result = script.add_active_line_indicators('say "hi"\n')
    assert 'log "##active_line1##"' in result


def test_applescript_preprocess_code_wraps_for_osascript():
    """User script is escaped and wrapped for osascript -e with execution markers."""
    script = AppleScript()
    result = script.preprocess_code('say "hello"')
    assert result.startswith("osascript -e ")
    assert "##end_of_execution##" in result
    assert '\\"hello\\"' in result


def test_applescript_detect_active_line():
    script = AppleScript()
    assert script.detect_active_line("##active_line4##") == 4


def test_applescript_detect_end_of_execution():
    script = AppleScript()
    assert script.detect_end_of_execution("##end_of_execution##")
