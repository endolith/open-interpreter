"""Real subprocess smokes for languages supported on Linux CI.

Unit tests mock preprocess/detect helpers; these catch hangs, missing binaries,
and marker parsing against a live interpreter process. Run in the Linux unit job
only (``linux_ci``); skipped locally on Windows/macOS when running the full suite.
"""

import shutil

import pytest

from interpreter import OpenInterpreter
from tests.helpers import console_output_text, require_bash_compatible_shell


@pytest.fixture
def interpreter():
    oi = OpenInterpreter()
    oi.conversation_history = False
    return oi


@pytest.mark.linux_ci
@pytest.mark.timeout(60)
def test_python_subprocess_smoke(interpreter):
    chunks = list(interpreter.computer.run("python", 'print("py_ok")'))
    assert "py_ok" in console_output_text(chunks)


@pytest.mark.linux_ci
@pytest.mark.timeout(30)
def test_javascript_subprocess_smoke(interpreter):
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    chunks = list(interpreter.computer.run("javascript", 'console.log("js_ok")'))
    assert "js_ok" in console_output_text(chunks)


@pytest.mark.linux_ci
@pytest.mark.timeout(30)
def test_shell_bash_echo_smoke(interpreter):
    require_bash_compatible_shell()
    chunks = list(interpreter.computer.run("shell", "echo shell_ok"))
    assert "shell_ok" in console_output_text(chunks)


@pytest.mark.linux_ci
@pytest.mark.timeout(30)
def test_shell_bash_nested_loop_quoting(interpreter):
    require_bash_compatible_shell()
    code = 'for i in a b; do for j in 1 2; do echo "${i}_${j}"; done; done'
    chunks = list(interpreter.computer.run("shell", code))
    output = console_output_text(chunks)
    assert "a_1" in output
    assert "b_2" in output


@pytest.mark.linux_ci
@pytest.mark.timeout(30)
def test_ruby_subprocess_smoke(interpreter):
    if shutil.which("irb") is None:
        pytest.fail("irb not found — Linux CI installs the ruby package")
    chunks = list(interpreter.computer.run("ruby", 'puts "ruby_ok"'))
    assert "ruby_ok" in console_output_text(chunks)


@pytest.mark.linux_ci
@pytest.mark.timeout(30)
def test_r_subprocess_smoke(interpreter):
    if shutil.which("R") is None:
        pytest.fail("R not found — Linux CI installs the r-base package")
    chunks = list(interpreter.computer.run("r", 'cat("r_ok\\n")'))
    assert "r_ok" in console_output_text(chunks)
