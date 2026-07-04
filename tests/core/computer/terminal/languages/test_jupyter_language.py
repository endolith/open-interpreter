import ast

from interpreter.core.computer.terminal.languages.jupyter_language import (
    AddLinePrints,
    add_active_line_prints,
    preprocess_python,
    string_to_python,
    wrap_in_try_except,
)


def test_preprocess_python_adds_active_line_markers():
    """preprocess_python() injects ##active_line markers into executable Python lines."""
    code = "x = 1\nprint(x)"
    result = preprocess_python(code)
    assert "##active_line" in result


def test_preprocess_python_skips_magic_lines():
    """preprocess_python() leaves IPython magic lines without active-line markers."""
    code = "%matplotlib inline\nx = 1"
    result = preprocess_python(code)
    assert "##active_line" not in result


def test_preprocess_python_respects_active_line_detection_env(monkeypatch):
    """INTERPRETER_ACTIVE_LINE_DETECTION=false disables marker injection."""
    monkeypatch.setenv("INTERPRETER_ACTIVE_LINE_DETECTION", "false")
    code = "x = 1"
    result = preprocess_python(code)
    assert "##active_line" not in result


def test_preprocess_python_strips_blank_lines(monkeypatch):
    """preprocess_python() removes blank lines when active-line detection is disabled."""
    monkeypatch.setenv("INTERPRETER_ACTIVE_LINE_DETECTION", "false")
    code = "x = 1\n\n\ny = 2"
    result = preprocess_python(code)
    assert result == "x = 1\ny = 2"


def test_add_active_line_prints_inserts_before_executable_lines():
    """add_active_line_prints() adds markers before each executable Python statement."""
    code = "a = 1\nprint(a)"
    result = add_active_line_prints(code)
    assert result.count("##active_line") >= 2


def test_add_active_line_prints_handles_comments():
    """add_active_line_prints() skips comment-only lines but still processes executable code."""
    code = "# comment\nx = 1"
    result = add_active_line_prints(code)
    assert "x = 1" in result


def test_string_to_python_extracts_public_functions():
    """string_to_python() returns public function definitions with their imports, excluding private names."""
    code = """
import os

def hello():
    \"\"\"Say hi\"\"\"
    return 1

def _private():
    pass
"""
    functions = string_to_python(code)
    assert "hello" in functions
    assert "_private" not in functions
    assert "def hello():" in functions["hello"]
    assert "import os" in functions["hello"]


def test_wrap_in_try_except_wraps_code():
    """wrap_in_try_except() wraps user code in try/except with traceback printing."""
    code = "x = 1"
    result = wrap_in_try_except(code)
    assert "try:" in result
    assert "traceback.print_exc" in result


def test_add_line_prints_transformer_inserts_prints():
    """AddLinePrints AST transformer injects ##active_line print statements into parsed code."""
    tree = ast.parse("x = 1")
    transformer = AddLinePrints()
    new_tree = transformer.visit(tree)
    unparsed = ast.unparse(new_tree)
    assert "##active_line" in unparsed
