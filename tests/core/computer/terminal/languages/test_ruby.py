import shutil
from unittest import mock

import pytest

from interpreter.core.terminal.languages.ruby import Ruby


def test_ruby_preprocess_code_wraps_in_begin_rescue():
    """Ruby code is wrapped in begin/rescue with active-line and end-of-execution markers."""
    ruby = Ruby()
    result = ruby.preprocess_code("puts 1")
    assert "begin" in result
    assert "##end_of_execution##" in result
    assert "##active_line1##" in result
    assert "rescue Exception" in result, "non-StandardError exceptions must not kill the run"


def test_ruby_line_postprocessor_skips_echoed_code():
    """Ruby line_postprocessor returns None while still within the submitted code line count."""
    ruby = Ruby()
    ruby.preprocess_code("puts 1")
    assert ruby.line_postprocessor("ignored") is None


def test_ruby_line_postprocessor_filters_nil():
    """Ruby line_postprocessor drops => nil result lines from output."""
    ruby = Ruby()
    ruby.code_line_count = 0
    assert ruby.line_postprocessor("=> nil") is None


def test_ruby_detect_active_line():
    """Ruby detect_active_line() parses ##active_lineN## markers and ignores plain text."""
    ruby = Ruby()
    assert ruby.detect_active_line("##active_line3##") == 3
    assert ruby.detect_active_line("hello") is None


def test_ruby_detect_end_of_execution():
    """Ruby detect_end_of_execution() recognizes end and error markers but not normal output."""
    ruby = Ruby()
    assert ruby.detect_end_of_execution("##end_of_execution##") is True
    assert ruby.detect_end_of_execution("##execution_error##") is True
    assert ruby.detect_end_of_execution("(SyntaxError)") is True
    assert ruby.detect_end_of_execution("syntax error, unexpected end") is False
    assert ruby.detect_end_of_execution("running") is False


def test_ruby_start_cmd_is_irb(monkeypatch):
    """With no resolvable launcher, Ruby keeps the bare ["irb"] name."""
    monkeypatch.setattr(
        "interpreter.core.terminal.languages.ruby.shutil.which",
        lambda _name: None,
    )
    assert Ruby().start_cmd == ["irb"]


def test_ruby_start_cmd_keeps_a_direct_launcher_on_posix(monkeypatch):
    """A launcher POSIX can execute itself must not be routed through cmd.exe."""
    monkeypatch.setattr(
        "interpreter.core.terminal.languages.ruby.shutil.which",
        lambda _name: "/usr/bin/irb",
    )
    monkeypatch.setattr("interpreter.core.terminal.languages.ruby.os.name", "posix")
    assert Ruby().start_cmd == ["irb"]


def test_ruby_start_cmd_wraps_a_batch_launcher_on_windows(monkeypatch):
    """Windows cannot run a .bat/.cmd launcher directly; cmd.exe must run it.

    Popen(["irb"]) fails with WinError 2 there because CreateProcess only
    loads .exe images, so the resolved launcher is handed to cmd.exe.
    """
    monkeypatch.setattr(
        "interpreter.core.terminal.languages.ruby.shutil.which",
        lambda _name: r"C:\Ruby\bin\irb.BAT",
    )
    monkeypatch.setattr("interpreter.core.terminal.languages.ruby.os.name", "nt")
    monkeypatch.delenv("COMSPEC", raising=False)
    assert Ruby().start_cmd == [
        "cmd.exe",
        "/d",
        "/s",
        "/c",
        r'""C:\Ruby\bin\irb.BAT""',
    ]


def test_ruby_start_cmd_quotes_a_launcher_with_cmd_metacharacters(monkeypatch):
    """A path with &, parentheses, or spaces survives cmd's command splitter.

    list2cmdline only quotes on whitespace and cmd strips the quotes it is
    given, so those characters would otherwise split the launch into extra
    commands; the inner quote pair is what keeps the path intact.
    """
    monkeypatch.setattr(
        "interpreter.core.terminal.languages.ruby.shutil.which",
        lambda _name: r"C:\Users\John (Work)\Ruby&Tools\bin\irb.cmd",
    )
    monkeypatch.setattr("interpreter.core.terminal.languages.ruby.os.name", "nt")
    monkeypatch.delenv("COMSPEC", raising=False)
    cmd = Ruby().start_cmd
    assert cmd[:4] == ["cmd.exe", "/d", "/s", "/c"]
    assert cmd[4] == r'""C:\Users\John (Work)\Ruby&Tools\bin\irb.cmd""'


def test_ruby_start_cmd_honors_comspec(monkeypatch):
    """The Windows command processor comes from COMSPEC when it is set."""
    monkeypatch.setattr(
        "interpreter.core.terminal.languages.ruby.shutil.which",
        lambda _name: r"C:\Ruby\bin\irb.cmd",
    )
    monkeypatch.setattr("interpreter.core.terminal.languages.ruby.os.name", "nt")
    monkeypatch.setenv("COMSPEC", r"D:\Windows\system32\cmd.exe")
    assert Ruby().start_cmd[0] == r"D:\Windows\system32\cmd.exe"


def test_ruby_terminate_kills_the_cmd_wrapper_tree_on_windows(monkeypatch):
    """Killing cmd.exe alone leaves the batch-launched irb running.

    cmd.exe does not terminate the program it launched, so terminate() must
    taskkill the tree before the base implementation reaps the wrapper.
    """
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return mock.Mock(returncode=0)

    monkeypatch.setattr(
        "interpreter.core.terminal.languages.ruby.subprocess.run", fake_run
    )
    monkeypatch.setattr("interpreter.core.terminal.languages.ruby.os.name", "nt")
    monkeypatch.setenv("SystemRoot", r"D:\Windows")
    ruby = Ruby()
    ruby.start_cmd = ["cmd.exe", "/d", "/s", "/c", r'""C:\Ruby\bin\irb.cmd""']
    ruby._launched_through_cmd = True
    ruby.process = mock.Mock(pid=4321)

    ruby.terminate()

    assert calls[0] == [
        r"D:\Windows\System32\taskkill.exe",
        "/T",
        "/F",
        "/PID",
        "4321",
    ]


def test_ruby_terminate_skips_taskkill_off_windows(monkeypatch):
    """POSIX has no cmd.exe wrapper, so terminate() must not shell out."""
    calls = []
    monkeypatch.setattr(
        "interpreter.core.terminal.languages.ruby.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )
    monkeypatch.setattr("interpreter.core.terminal.languages.ruby.os.name", "posix")
    ruby = Ruby()
    ruby.start_cmd = ["cmd.exe", "/d", "/s", "/c", r'""x""']
    ruby._launched_through_cmd = True
    ruby.process = mock.Mock(pid=4321)

    ruby.terminate()

    assert calls == []


def test_ruby_terminate_skips_taskkill_for_the_bare_irb_form(monkeypatch):
    """The plain ["irb"] process is irb itself; there is no wrapper to kill."""
    calls = []
    monkeypatch.setattr(
        "interpreter.core.terminal.languages.ruby.shutil.which", lambda _name: None
    )
    monkeypatch.setattr(
        "interpreter.core.terminal.languages.ruby.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )
    monkeypatch.setattr("interpreter.core.terminal.languages.ruby.os.name", "nt")
    ruby = Ruby()
    ruby.process = mock.Mock(pid=4321)

    ruby.terminate()

    assert calls == []


def test_ruby_detect_end_of_execution_covers_error_marker():
    """The Ruby end detector treats both the end and error markers as completion.

    The rescue path emits ##execution_error## instead of ##end_of_execution##;
    the detector must accept either or errored scripts would never complete.
    """
    ruby = Ruby()
    assert ruby.detect_end_of_execution("##execution_error##") is True
    assert ruby.detect_end_of_execution("##end_of_execution##") is True


def test_ruby_line_postprocessor_consumes_exactly_the_code_line_count():
    """Each postprocess call decrements code_line_count by exactly one.

    The counter is set from the wrapped code and consumed one line per call; a
    -= 2 mutation would skip real output lines after the echoed code.
    """
    ruby = Ruby()
    ruby.preprocess_code("puts 1")
    start = ruby.code_line_count

    ruby.line_postprocessor("first")
    assert ruby.code_line_count == start - 1

    ruby.line_postprocessor("second")
    assert ruby.code_line_count == start - 2


def test_ruby_end_marker_survives_a_drifted_echo_window():
    """An exact end marker is passed and closes the skip window even mid-count.

    irb echoes more lines than the wrapped block contains, so the line counter
    is still positive when the real marker arrives on later calls. Skipping it
    would leave the run waiting forever; the marker is control channel, not
    echoed source, so the postprocessor must let it through and stop counting.
    """
    ruby = Ruby()
    ruby.preprocess_code("1+1")
    assert ruby.code_line_count > 0

    line = ruby.line_postprocessor("##end_of_execution##\n")

    assert line == "##end_of_execution##\n"
    assert ruby.code_line_count == 0


def test_ruby_active_line_marker_survives_a_drifted_echo_window():
    """An exact active-line marker also bypasses the skip window.

    The same drift that swallowed the end marker dropped the highlight marker,
    so later calls showed no active line even though execution succeeded.
    """
    ruby = Ruby()
    ruby.preprocess_code("1+1")
    assert ruby.code_line_count > 0

    line = ruby.line_postprocessor("##active_line1##\n")

    assert line == "##active_line1##\n"
    assert ruby.code_line_count == 0


def test_ruby_echoed_marker_source_is_still_skipped():
    """The echoed `puts "##end_of_execution##"` line is not treated as the marker.

    If the exemption matched echoed source, irb would end the block while it is
    still reading the input, long before the program runs.
    """
    ruby = Ruby()
    ruby.preprocess_code("1+1")
    assert ruby.code_line_count > 0

    assert ruby.line_postprocessor('    puts "##end_of_execution##\\n"\n') is None


def test_ruby_nil_substring_is_real_output():
    """Only a bare nil result is filtered, not values that contain `nil`."""
    ruby = Ruby()
    ruby.code_line_count = 0

    assert ruby.line_postprocessor("vanilla\n") == "vanilla\n"
    assert ruby.line_postprocessor("nilpotent\n") == "nilpotent\n"
    assert ruby.line_postprocessor("nil\n") is None


def test_ruby_echo_window_counts_echoed_lines_not_split_segments(monkeypatch):
    """The window spans exactly the lines irb echoes, not the final newline too.

    split("\n") returns a phantom trailing element for the wrapper's terminating
    newline, so the window stayed open one line past the echo. The first block's
    session banner happened to spend that extra line; every later block spent it
    on real output.
    """
    monkeypatch.setenv("INTERPRETER_ACTIVE_LINE_DETECTION", "false")
    ruby = Ruby()
    processed = ruby.preprocess_code('puts "C"; puts "D"')
    assert ruby.code_line_count == len(processed.rstrip("\n").split("\n"))


def test_ruby_session_banner_does_not_consume_a_source_line():
    """irb's one-time banner is consumed without decrementing the window.

    If it counted as an echoed line, the window would close before the last
    echoed source line and leak that line into the output.
    """
    ruby = Ruby()
    ruby.preprocess_code("1+1")
    start = ruby.code_line_count
    assert ruby.line_postprocessor("Switch to inspect mode.\n") is None
    assert ruby.code_line_count == start


def test_ruby_echo_window_closes_before_the_first_output_line(monkeypatch):
    """After the echoed block, the first real output line survives.

    Feeds the echoed block lines only (no banner, as on later blocks), then the
    first output line; the window must already be closed.
    """
    monkeypatch.setenv("INTERPRETER_ACTIVE_LINE_DETECTION", "false")
    ruby = Ruby()
    processed = ruby.preprocess_code('puts "C"; puts "D"')
    for raw in processed.splitlines():
        assert ruby.line_postprocessor(raw + "\n") is None
    assert ruby.code_line_count == 0
    assert ruby.line_postprocessor("C\n") == "C\n"


def test_ruby_later_blocks_keep_their_first_output_line(monkeypatch):
    """Live irb: the second block in a session still prints both its lines.

    The bug only appeared after the one-time banner was gone, so a single-block
    test cannot catch it.
    """
    if shutil.which("irb") is None:
        pytest.skip("irb not installed; Ruby REPL coverage needs it")
    monkeypatch.setenv("INTERPRETER_ACTIVE_LINE_DETECTION", "false")
    ruby = Ruby()
    ruby.start_process()
    try:
        output = []
        for code in ('puts "A"; puts "B"', 'puts "C"; puts "D"'):
            output.append(
                "".join(
                    str(chunk["content"])
                    for chunk in ruby.run(code)
                    if chunk.get("format") == "output"
                )
            )
    finally:
        ruby.terminate()
    assert "A" in output[0] and "B" in output[0]
    assert "C" in output[1] and "D" in output[1]
