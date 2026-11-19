from interpreter.core.terminal.languages.powershell import (
    PowerShell,
    preprocess_powershell,
)


def test_preprocess_powershell_adds_end_marker():
    """preprocess_powershell() appends an ##end_of_execution## marker to PowerShell code."""
    code = preprocess_powershell("Write-Host hi")
    assert "##end_of_execution##" in code


def test_powershell_detect_active_line():
    """PowerShell detect_active_line() parses ##active_lineN## markers from echoed output."""
    # Bypass __init__: resolving pwsh is not needed to parse a marker line,
    # and these tests run on hosts without PowerShell installed.
    ps = PowerShell.__new__(PowerShell)
    assert ps.detect_active_line('echo "##active_line4##"') == 4


def _bare_powershell(cwd="/tmp"):
    ps = PowerShell.__new__(PowerShell)
    ps.cwd = cwd
    ps._last_exit_status = None
    return ps


def test_powershell_linux_prompt_echo_is_filtered():
    """`PS /home/user>` is a prompt echo on Unix, not program output.

    The old regex only accepted Windows drive-letter prompts, so Linux runs
    printed their own prompt and the echoed source line into the result.
    """
    ps = _bare_powershell()

    assert ps.line_postprocessor('PS /home/user> Write-Output "hi"\n') is None
    assert ps.line_postprocessor("(base) PS /home/user> 1+1\n") is None


def test_powershell_windows_prompt_echo_is_still_filtered():
    """The platform-agnostic prompt match keeps the Windows behaviour."""
    ps = _bare_powershell()

    assert ps.line_postprocessor('PS C:\\Users\\user> Write-Output "hi"\n') is None
    assert ps.line_postprocessor("PS D:\\work> ") is None


def test_powershell_continuation_echo_with_ansi_escape_is_filtered():
    """A continuation echo prefixed by pwsh's DECCKM escape is dropped.

    pwsh emits ESC[?1l before the line, so `>> ` is not at column 0 and the old
    startswith check let the echoed code through.
    """
    ps = _bare_powershell()

    assert ps.line_postprocessor('\x1b[?1l>> \x1b[?1h  $ErrorActionPreference = "Stop"\n') is None
    assert ps.line_postprocessor("\x1b[?1l>>\x1b[?1h\n") is None


def test_powershell_echoed_cwd_marker_source_is_not_parsed_as_cwd():
    """The echoed `Write-Output "##oi_cwd##..."` line must not update cwd.

    Marker parsing runs before the display filter in the mixin. If the echo
    reaches it, the literal `$($PWD.Path)"` becomes the tracked directory and
    the echoed tail leaks into output.
    """
    ps = _bare_powershell(cwd="/home/user")

    result = ps.line_postprocessor('PS /home/user> \x1b[?1hWrite-Output "##oi_cwd##$($PWD.Path)"\n')

    assert result is None
    assert ps.cwd == "/home/user"


def test_powershell_ansi_prefixed_cwd_marker_still_updates_cwd():
    """The real marker arrives after the escape and is still consumed."""
    ps = _bare_powershell(cwd="/home/user")

    assert ps.line_postprocessor("\x1b[?1l##oi_cwd##/home/user/src\n") is None
    assert ps.cwd == "/home/user/src"


def test_powershell_real_output_with_ansi_prefix_is_kept_clean():
    """Real program output carries the same escape prefix but must survive."""
    ps = _bare_powershell()

    assert ps.line_postprocessor("\x1b[?1lhello\n") == "hello\n"


def test_powershell_end_marker_with_ansi_prefix_is_detected():
    """The end marker reaches detect_end_of_execution even after an escape."""
    ps = _bare_powershell()

    line = ps.line_postprocessor("\x1b[?1l##end_of_execution##\n")

    assert line == "##end_of_execution##\n"
    assert ps.detect_end_of_execution(line) is True


def test_powershell_output_without_trailing_newline_keeps_output_before_prompt():
    """Output lacking a final newline is kept; the prompt drawn after it is not.

    pwsh writes the next prompt on the same line as output that did not end in
    a newline, so `xPS /home/user> ` used to leak the whole line. The ESC[?1h
    that introduces the echoed input marks where program output ends.
    """
    ps = _bare_powershell(cwd="/home/user")

    result = ps.line_postprocessor('xPS /home/user> \x1b[?1hWrite-Output "##oi_cwd##$($PWD.Path)"\n')

    assert result == "x"
    assert ps.cwd == "/home/user"


def test_powershell_conda_prompt_after_output_keeps_output():
    """A conda-prefixed prompt glued to output is removed, output survives."""
    ps = _bare_powershell(cwd="/home/user")

    result = ps.line_postprocessor('done(base) PS /home/user> \x1b[?1hWrite-Output "hi"\n')

    assert result == "done"


def test_powershell_echoed_input_without_prompt_is_dropped():
    """Every echoed input line carries ESC[?1h, even without a visible prompt."""
    ps = _bare_powershell(cwd="/home/user")

    assert ps.line_postprocessor('\x1b[?1hWrite-Host "hi"\n') is None
    assert ps.line_postprocessor('\x1b[?1l>> \x1b[?1h  $ErrorActionPreference = "Stop"\n') is None


def test_powershell_blank_prompt_before_echo_is_dropped():
    """A prompt followed only by the echo marker produces no output line."""
    ps = _bare_powershell(cwd="/home/user")

    assert ps.line_postprocessor("PS /home/user> \x1b[?1h\n") is None
