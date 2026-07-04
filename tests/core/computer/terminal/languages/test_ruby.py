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
