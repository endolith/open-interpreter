from interpreter.core.terminal.languages.r import R


def test_r_preprocess_code_wraps_in_trycatch():
    """R code is wrapped in tryCatch with active-line and end-of-execution markers."""
    r_lang = R()
    result = r_lang.preprocess_code("x <- 1")
    assert "tryCatch" in result
    assert "##end_of_execution##" in result
    assert "##active_line1##" in result


def test_r_line_postprocessor_does_not_skip_real_output():
    """--no-echo means no echoed program exists; ordinary output and the end marker survive.

    Skipping a fixed line count here swallowed real output (including
    ##end_of_execution##) and hung every R run.
    """
    r_lang = R()
    r_lang.preprocess_code("x <- 1")
    assert r_lang.line_postprocessor("ordinary output") == "ordinary output"
    assert r_lang.line_postprocessor("##end_of_execution##") is not None


def test_r_line_postprocessor_strips_string_output():
    """R line_postprocessor strips the [1] prefix and quotes from string output."""
    r_lang = R()
    assert r_lang.line_postprocessor('[1] "hello"') == "hello"


def test_r_line_postprocessor_strips_numeric_prefix():
    """R line_postprocessor strips the [1] prefix from numeric output."""
    r_lang = R()
    assert r_lang.line_postprocessor("[1] 42") == "42"


def test_r_line_postprocessor_filters_prompt_lines():
    """R line_postprocessor drops bare R prompt lines from output."""
    r_lang = R()
    assert r_lang.line_postprocessor("> ") is None


def test_r_line_postprocessor_keeps_version_output():
    """`cat(R.version.string)` is real output and must not be mistaken for the banner.

    R is started with -q (no banner), so dropping any line containing
    "R version" swallowed legitimate user output.
    """
    r_lang = R()
    assert r_lang.line_postprocessor("R version 4.5.2 (2025-10-31) ok") is not None


def test_r_detect_active_line():
    """R detect_active_line() parses the line number from ##active_lineN## markers."""
    r_lang = R()
    assert r_lang.detect_active_line("##active_line2##") == 2


def test_r_detect_end_of_execution():
    """R detect_end_of_execution() recognizes end and error markers but not normal output."""
    r_lang = R()
    assert r_lang.detect_end_of_execution("##end_of_execution##") is True
    assert r_lang.detect_end_of_execution("##execution_error##") is True
    assert r_lang.detect_end_of_execution("still running") is False
