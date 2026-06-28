from interpreter.core.computer.terminal.languages.r import R


def test_r_preprocess_code_wraps_in_trycatch():
    """R code is wrapped in tryCatch with active-line and end-of-execution markers."""
    r_lang = R()
    result = r_lang.preprocess_code("x <- 1")
    assert "tryCatch" in result
    assert "##end_of_execution##" in result
    assert "##active_line1##" in result


def test_r_line_postprocessor_skips_echoed_code():
    r_lang = R()
    r_lang.preprocess_code("x <- 1")
    assert r_lang.line_postprocessor("ignored") is None


def test_r_line_postprocessor_strips_string_output():
    r_lang = R()
    r_lang.code_line_count = 0
    assert r_lang.line_postprocessor('[1] "hello"') == "hello"


def test_r_line_postprocessor_strips_numeric_prefix():
    r_lang = R()
    r_lang.code_line_count = 0
    assert r_lang.line_postprocessor("[1] 42") == "42"


def test_r_line_postprocessor_filters_prompt_lines():
    r_lang = R()
    r_lang.code_line_count = 0
    assert r_lang.line_postprocessor("> ") is None


def test_r_detect_active_line():
    r_lang = R()
    assert r_lang.detect_active_line("##active_line2##") == 2


def test_r_detect_end_of_execution():
    r_lang = R()
    assert r_lang.detect_end_of_execution("##end_of_execution##") is True
    assert r_lang.detect_end_of_execution("##execution_error##") is True
    assert r_lang.detect_end_of_execution("still running") is False
