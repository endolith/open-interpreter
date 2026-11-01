import ast
from unittest import mock

import interpreter.core.terminal.languages.jupyter_language as jl
from interpreter.core.terminal.languages.jupyter_language import (
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


def test_start_kernel_mutes_the_loopback_tcp_warning(monkeypatch):
    """The kernel starts at --log-level=ERROR so its unencrypted-TCP warning is not shown."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.km = mock.Mock()
    monkeypatch.setattr(jl.sys, "platform", "linux")

    lang._start_kernel()

    lang.km.start_kernel.assert_called_once_with(
        extra_arguments=["--log-level=ERROR"]
    )


def test_kernel_is_alive_uses_the_process_not_the_client():
    """The client can stay up after the process exits; liveness must use the manager."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False
    assert lang._kernel_is_alive() is False

    lang.km.is_alive.return_value = True
    assert lang._kernel_is_alive() is True


def test_python_recovers_after_the_kernel_hard_exits():
    """os._exit(0) must end the run, and the next run must restart the kernel.

    A regression here hangs the REPL: the kernel is dead, the client channels
    stay open, and the old run waited forever for an idle message.
    """
    import threading
    import types

    from interpreter.core.terminal.languages.python import Python

    def collect(generator, timeout=30):
        result = {}

        def worker():
            try:
                result["chunks"] = list(generator)
            except Exception as exc:  # pragma: no cover - surfaced by the assertion
                result["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout)
        assert not thread.is_alive(), "run() hung instead of returning"
        assert "error" not in result, result.get("error")
        return result["chunks"]

    interpreter = types.SimpleNamespace(
        stop_event=threading.Event(),
        messages=[],
        verbose=False,
        debug=False,
        max_output=2000,
        plain_text_display=True,
    )
    language = Python(interpreter)
    try:
        collect(language.run("print(1)"))
        killed = collect(
            language.run("import os, sys; sys.stdout.flush(); os._exit(0)")
        )
        assert any(
            "kernel exited" in str(chunk.get("content", "")) for chunk in killed
        )
        revived = collect(language.run("print(2)"))
        assert any("2" in str(chunk.get("content", "")) for chunk in revived)
    finally:
        language.terminate()
