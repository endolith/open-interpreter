from interpreter.core.computer.terminal.languages.applescript import AppleScript


def test_applescript_add_active_line_indicators():
    """AppleScript add_active_line_indicators() inserts log markers before each line."""
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
    """AppleScript detect_active_line() parses the line number from ##active_lineN## markers."""
    script = AppleScript()
    assert script.detect_active_line("##active_line4##") == 4


def test_applescript_detect_end_of_execution():
    """AppleScript detect_end_of_execution() recognizes end markers but not running output."""
    script = AppleScript()
    assert script.detect_end_of_execution("##end_of_execution##") is True
    assert script.detect_end_of_execution("still running") is False
