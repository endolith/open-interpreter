from interpreter.core.computer.terminal.languages.javascript import (
    JavaScript,
    preprocess_javascript,
)


def test_preprocess_javascript_adds_end_marker():
    """preprocess_javascript() wraps code in try/catch and adds an end-of-execution marker."""
    code = preprocess_javascript("console.log(1)")
    assert "##end_of_execution##" in code
    assert "try {" in code


def test_preprocess_single_line_adds_active_line_markers():
    """preprocess_javascript() inserts ##active_lineN## markers before each source line."""
    code = preprocess_javascript("a()\nb()")
    assert "##active_line1##" in code
    assert "##active_line2##" in code


def test_line_postprocessor_filters_node_banner():
    """JavaScript line_postprocessor drops Node.js welcome banners but keeps real output."""
    js = JavaScript()
    assert js.line_postprocessor("Welcome to Node.js v20") is None
    assert js.line_postprocessor("actual output") == "actual output"
