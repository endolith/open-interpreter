from io import StringIO

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


def test_ruby_preprocess_rescues_exceptions_but_re_raises_signals():
    """The wrapper reports explicit Exception but never swallows signals or exits.

    `rescue Exception` alone would capture Ctrl-C and process exits, turning an
    interrupted run into an execution error. Re-raising those two first keeps
    their behavior while still printing the marker for other Exception
    subclasses.
    """
    ruby = Ruby()
    code = ruby.preprocess_code("raise Exception")
    assert "rescue SignalException, SystemExit" in code
    assert "rescue Exception => e" in code
    assert code.index("rescue SignalException") < code.index("rescue Exception")


def test_ruby_detect_end_of_execution_covers_syntax_errors():
    """A parser error ends the run even when the wrapper never prints a marker."""
    ruby = Ruby()
    assert ruby.detect_end_of_execution("(SyntaxError)") is False
    assert ruby.detect_error_end_of_execution("(SyntaxError)") is True
    assert ruby.detect_error_end_of_execution("the program printed syntax error") is False


def test_ruby_syntax_error_ends_on_either_stream():
    """irb may print a parser error to stdout (Ruby 3.2) or stderr; either ends it."""
    stdout_ruby = Ruby()
    stdout_ruby.handle_stream_output(StringIO("(SyntaxError)\n"), is_error_stream=False)
    assert stdout_ruby.done.is_set()

    stderr_ruby = Ruby()
    stderr_ruby.handle_stream_output(StringIO("(SyntaxError)\n"), is_error_stream=True)
    assert stderr_ruby.done.is_set()
