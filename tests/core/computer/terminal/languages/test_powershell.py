from interpreter.core.computer.terminal.languages.powershell import (
    PowerShell,
    preprocess_powershell,
)


def test_preprocess_powershell_adds_end_marker():
    """preprocess_powershell() appends an ##end_of_execution## marker to PowerShell code."""
    code = preprocess_powershell("Write-Host hi")
    assert "##end_of_execution##" in code


def test_powershell_detect_active_line():
    """PowerShell detect_active_line() parses ##active_lineN## markers from echoed output."""
    ps = PowerShell()
    assert ps.detect_active_line('echo "##active_line4##"') == 4
