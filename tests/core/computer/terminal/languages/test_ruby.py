import shutil

import pytest

from interpreter.core.computer.terminal.languages.ruby import Ruby


def test_ruby_preprocess_code_wraps_in_begin_rescue():
    """Ruby code is wrapped in begin/rescue with active-line and end-of-execution markers."""
    ruby = Ruby()
    result = ruby.preprocess_code("puts 1")
    assert "begin" in result
    assert "##end_of_execution##" in result
    assert "##active_line1##" in result


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
    assert ruby.detect_end_of_execution("running") is False


def test_ruby_start_cmd_is_irb():
    """Ruby starts the irb REPL via ["irb"]."""
    assert Ruby().start_cmd == ["irb"]


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
    """An exact end marker is passed through even while the skip window is open.

    irb echoes more lines than the wrapped block contains (banner, blanks,
    wrapped lines), so the counter can still be positive when the real end
    marker arrives. Skipping it leaves run() waiting forever for a completion
    signal that was already consumed, so the postprocessor must let exact
    control markers through and close the window.
    """
    ruby = Ruby()
    ruby.preprocess_code("1+1")
    assert ruby.code_line_count > 0

    line = ruby.line_postprocessor("##end_of_execution##\n")

    assert line == "##end_of_execution##\n"
    assert ruby.code_line_count == 0


def test_ruby_active_line_marker_survives_a_drifted_echo_window():
    """An exact active-line marker bypasses the skip window and resets it.

    The same drift that swallowed the end marker also dropped the highlight
    marker, so later runs showed no active line even while executing.
    """
    ruby = Ruby()
    ruby.preprocess_code("1+1")
    assert ruby.code_line_count > 0

    line = ruby.line_postprocessor("##active_line1##\n")

    assert line == "##active_line1##\n"
    assert ruby.code_line_count == 0


def test_ruby_echoed_marker_source_is_still_skipped():
    """An echoed `puts "##end_of_execution##"` line stays inside the skip window.

    irb echoes the submitted source, which contains the marker as a string.
    Treating that echo as the real marker would end the block while irb is
    still reading the input, before the program runs.
    """
    ruby = Ruby()
    ruby.preprocess_code("1+1")
    assert ruby.code_line_count > 0

    assert ruby.line_postprocessor('    puts "##end_of_execution##\\n"\n') is None


def test_ruby_nil_substring_is_real_output():
    """Only a bare nil result is filtered, not output that contains `nil`.

    The old substring test discarded real values such as "vanilla" and
    "nilpotent" because they contain the letters n-i-l.
    """
    ruby = Ruby()
    ruby.code_line_count = 0

    assert ruby.line_postprocessor("vanilla\n") == "vanilla\n"
    assert ruby.line_postprocessor("nilpotent\n") == "nilpotent\n"
    assert ruby.line_postprocessor("nil\n") is None
    assert ruby.line_postprocessor("=> nil\n") is None


def test_ruby_echo_window_counts_echoed_lines_not_split_segments():
    """The skip window spans exactly the lines irb echoes back.

    split("\n") yields a phantom trailing element for the wrapper's final
    newline, so the window stayed open one line past the echo. On the first
    block irb's session banner spent that extra line; on later blocks it ate
    the block's first real output line.
    """
    ruby = Ruby()
    processed = ruby.preprocess_code('puts "C"; puts "D"')

    assert ruby.code_line_count == len(processed.rstrip("\n").split("\n"))
    assert ruby.code_line_count == len(processed.splitlines())


def test_ruby_session_banner_does_not_consume_a_source_line():
    """irb's one-time banner is consumed without spending an echoed line.

    If the banner counted as an echoed line, the window would close before the
    last echoed source line and leak that line into the output as text.
    """
    ruby = Ruby()
    ruby.preprocess_code("1+1")
    start = ruby.code_line_count

    assert ruby.line_postprocessor("Switch to inspect mode.\n") is None
    assert ruby.code_line_count == start


def test_ruby_echo_window_closes_before_the_first_output_line():
    """After the echoed block, the first real output line is not swallowed.

    Feeds every echoed line (as irb prints them on a block after the banner)
    and then the first output line; the window must already be closed.
    """
    ruby = Ruby()
    processed = ruby.preprocess_code('puts "C"; puts "D"')
    for raw in processed.splitlines():
        assert ruby.line_postprocessor(raw + "\n") is None

    assert ruby.code_line_count == 0
    assert ruby.line_postprocessor("C\n") == "C\n"


def test_ruby_later_blocks_keep_their_first_output_line():
    """Live irb: a session's second block still prints both its output lines.

    The skipped line only appeared after the one-time banner was gone, so a
    single-block test cannot catch it.
    """
    if shutil.which("irb") is None:
        pytest.skip("irb not installed; Ruby REPL coverage needs it")

    ruby = Ruby()
    try:
        ruby.start_process()
    except OSError as exc:
        # On Windows CI `shutil.which("irb")` can resolve a shim that Popen
        # still cannot launch; an absent REPL is a skip, not a failure.
        pytest.skip(f"irb could not be started: {exc}")
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
