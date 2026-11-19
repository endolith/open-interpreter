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
