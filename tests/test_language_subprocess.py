"""Real runtime smokes for all languages exercised on Linux CI.

Unit tests mock preprocess/detect helpers; these catch hangs, missing binaries,
and marker parsing against a live interpreter process. All snippets are hardcoded
(not LLM-generated). For tests that call an LLM and execute whatever it returns,
see ``test_interpreter.py`` integration tests (require OPENAI_API_KEY).

All ten Terminal languages are covered across CI runners:

| Language    | CI runner |
|-------------|-----------|
| Python      | Linux     |
| Shell/bash  | Linux     |
| JavaScript  | Linux     |
| Ruby        | Linux     |
| R           | Linux     |
| Java        | Linux     |
| HTML        | Linux     |
| React       | Linux     |
| Shell/cmd   | Windows   |
| PowerShell  | Windows   |
| AppleScript | macOS     |

To run locally on Linux, install the language runtimes that are missing:

    sudo apt install nodejs ruby r-base default-jdk
    # For HTML/React (needs headless Chrome):
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
    sudo apt install ./google-chrome-stable_current_amd64.deb
"""

import shutil

import pytest

from interpreter import OpenInterpreter
from tests.helpers import (
    chunks_of_type,
    console_output_text,
    require_bash_compatible_shell,
    require_chrome_for_html,
)

pytestmark = pytest.mark.linux_ci

_JAVA_SMOKE = """class JavaOk {
    public static void main(String[] args) {
        System.out.println("java_ok");
    }
}"""

_REACT_SMOKE = """function App() {
  return <div>react_ok</div>;
}
ReactDOM.render(<App />, document.getElementById('root'));"""


@pytest.fixture
def interpreter():
    oi = OpenInterpreter()
    oi.conversation_history = False
    return oi


@pytest.mark.timeout(60)
def test_python_subprocess_smoke(interpreter):
    chunks = list(interpreter.computer.run("python", 'print("py_ok")'))
    assert "py_ok" in console_output_text(chunks)


@pytest.mark.timeout(30)
def test_javascript_subprocess_smoke(interpreter):
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    chunks = list(interpreter.computer.run("javascript", 'console.log("js_ok")'))
    assert "js_ok" in console_output_text(chunks)


@pytest.mark.timeout(30)
def test_shell_bash_echo_smoke(interpreter):
    require_bash_compatible_shell()
    chunks = list(interpreter.computer.run("shell", "echo shell_ok"))
    assert "shell_ok" in console_output_text(chunks)


@pytest.mark.timeout(30)
def test_shell_bash_nested_loop_quoting(interpreter):
    require_bash_compatible_shell()
    code = 'for i in a b; do for j in 1 2; do echo "${i}_${j}"; done; done'
    chunks = list(interpreter.computer.run("shell", code))
    output = console_output_text(chunks)
    assert "a_1" in output
    assert "b_2" in output


@pytest.mark.timeout(30)
def test_ruby_subprocess_smoke(interpreter):
    if shutil.which("irb") is None:
        pytest.skip("irb not installed")
    chunks = list(interpreter.computer.run("ruby", 'puts "ruby_ok"'))
    assert "ruby_ok" in console_output_text(chunks)


@pytest.mark.timeout(30)
def test_r_subprocess_smoke(interpreter):
    if shutil.which("R") is None:
        pytest.skip("R not installed")
    chunks = list(interpreter.computer.run("r", 'cat("r_ok\\n")'))
    assert "r_ok" in console_output_text(chunks)


@pytest.mark.timeout(60)
def test_java_subprocess_smoke(interpreter):
    if shutil.which("javac") is None or shutil.which("java") is None:
        pytest.skip("jdk not installed")
    chunks = list(interpreter.computer.run("java", _JAVA_SMOKE))
    assert "java_ok" in console_output_text(chunks)


@pytest.mark.timeout(90)
def test_html_runtime_smoke(interpreter):
    require_chrome_for_html()
    try:
        chunks = list(
            interpreter.computer.run(
                "html", '<html><body style="font-size:32px">html_ok</body></html>'
            )
        )
    except FileNotFoundError as exc:
        pytest.skip(f"html2image screenshot failed: {exc}")
    assert chunks_of_type(chunks, "console")
    assert chunks_of_type(chunks, "code")[0]["format"] == "html"
    images = chunks_of_type(chunks, "image")
    assert len(images) == 1
    assert images[0]["format"] == "base64.png"
    assert len(images[0]["content"]) > 50


@pytest.mark.timeout(120)
def test_react_runtime_smoke(interpreter):
    require_chrome_for_html()
    try:
        chunks = list(interpreter.computer.run("react", _REACT_SMOKE))
    except FileNotFoundError as exc:
        pytest.skip(f"html2image screenshot failed: {exc}")
    assert chunks_of_type(chunks, "console")
    assert chunks_of_type(chunks, "code")[0]["format"] == "html"
    images = chunks_of_type(chunks, "image")
    assert len(images) == 1
    assert images[0]["format"] == "base64.png"
    assert len(images[0]["content"]) > 50
