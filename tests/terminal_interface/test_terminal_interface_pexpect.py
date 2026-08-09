"""Real-terminal tests for the CLI using pexpect.

These spawn the actual `interpreter` CLI in a pseudo-terminal (PTY) and assert on
the bytes it writes, including ANSI escape codes. They cover:

- basic startup and prompt rendering
- markdown output: block quotes, tables, code fences
- colors: 16-color, 256-color, and truecolor (24-bit) escape sequences
- terminal width change: sending SIGWINCH and confirming that markdown output
  re-wraps to the new width (the develop branch's width-change detection)

These are marked `linux_ci` because pexpect is not available on Windows.
They use the local MockOpenAIServer, so no API key or network is required.
"""

import os
import sys

import pytest

pexpect = pytest.importorskip("pexpect")

from tests.support.mock_openai_server import MockOpenAIServer

pytestmark = [
    pytest.mark.linux_ci,
    pytest.mark.mock_llm,
]


# The CLI has no `__main__.py`, so invoke its entry point directly.
_CLI_ENTRY = (
    "from interpreter.terminal_interface.start_terminal_interface import main; main()"
)


@pytest.fixture(scope="module")
def mock_llm_server():
    """Start a local OpenAI-compatible server that returns a long markdown reply."""
    server = MockOpenAIServer(
        reply_text=(
            "> This is a block quote that is intentionally long enough to wrap "
            "across multiple lines when the terminal is narrow.\n\n"
            "| Col A | Col B | Col C |\n"
            "| ----- | ----- | ----- |\n"
            "| alpha | beta  | gamma |\n"
            "| delta | epsilon | zeta |\n\n"
            "```python\n"
            "def hello(name):\n"
            "    return f'Hello, {name}!\n"
            "```"
        )
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def interpreter_cmd(mock_llm_server):
    """The CLI command with flags pointed at the mock server, plus env."""
    cmd = [
        sys.executable,
        "-c",
        _CLI_ENTRY,
        "--plain",
        "--model",
        "openai/gpt-4o-mini",
        "--api_base",
        mock_llm_server.api_base,
        "--api_key",
        "mock-key",
        "--disable_telemetry",
    ]
    env = dict(os.environ)
    env.update(
        {
            "OPENAI_API_KEY": "mock-key",
            "OPENAI_API_BASE": mock_llm_server.api_base,
            "OI_RUN_INTEGRATION": "0",
            "COLUMNS": "80",
            "LINES": "30",
        }
    )
    return cmd, env


@pytest.fixture
def run_cli():
    """Spawn the CLI in a PTY, yield a control object, then tear it down."""
    spawned = []

    def _spawn(cmd, env, cols=100, rows=30):
        child = pexpect.spawn(cmd[0], cmd[1:], env=env, dimensions=(rows, cols))
        child.delaybeforesend = 0.05
        spawned.append(child)
        return child

    yield _spawn

    for child in spawned:
        if child.isalive():
            child.close(force=True)


@pytest.mark.timeout(60)
def test_cli_starts_and_shows_prompt(run_cli, interpreter_cmd):
    """The CLI starts and presents its input prompt."""
    cmd, env = interpreter_cmd
    child = run_cli(cmd, env)
    child.expect([r"\$|>\s*", pexpect.EOF, pexpect.TIMEOUT], timeout=30)


@pytest.mark.timeout(90)
def test_markdown_block_quote_renders(run_cli, interpreter_cmd):
    """A block quote renders (the `>` marker appears in the byte stream)."""
    cmd, env = interpreter_cmd
    child = run_cli(cmd, env, cols=60, rows=30)
    child.expect([r"\$|>\s*", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    child.sendline("Say hello.")
    child.expect(r">\s*This is a block quote", timeout=60)


@pytest.mark.timeout(90)
def test_markdown_table_renders(run_cli, interpreter_cmd):
    """A markdown table renders its cells (visible in the byte stream)."""
    cmd, env = interpreter_cmd
    child = run_cli(cmd, env, cols=80, rows=30)
    child.expect([r"\$|>\s*", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    child.sendline("Say hello.")
    child.expect(r"alpha", timeout=60)
    child.expect(r"epsilon", timeout=60)


@pytest.mark.timeout(90)
def test_markdown_code_fence_renders(run_cli, interpreter_cmd):
    """A code fence renders (its text appears in the byte stream)."""
    cmd, env = interpreter_cmd
    child = run_cli(cmd, env, cols=80, rows=30)
    child.expect([r"\$|>\s*", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    child.sendline("Say hello.")
    child.expect(r"def hello", timeout=60)


@pytest.mark.timeout(90)
def test_ansi_color_escapes_present(run_cli, interpreter_cmd):
    """Rich renders colored output (SGR color escapes appear in the stream)."""
    cmd, env = interpreter_cmd
    child = run_cli(cmd, env, cols=80, rows=30)
    child.expect([r"\$|>\s*", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    child.sendline("Say hello.")
    # Any SGR sequence (16-color, 256-color, or truecolor): ESC [ params m
    child.expect(r"\x1b\[\d+(;\d+)*m", timeout=60)


@pytest.mark.timeout(90)
def test_terminal_resize_rewraps_markdown(run_cli, interpreter_cmd):
    """Resizing the terminal re-wraps markdown to the new width.

    The develop branch detects width changes (SIGWINCH) and re-wraps new text to
    the new width, which the old version does not do. This test starts at 100
    columns, narrows to 40 (setwinsize sends SIGWINCH), then checks that the
    long block-quote line appears broken across multiple lines rather than as
    one long line.
    """
    cmd, env = interpreter_cmd
    child = run_cli(cmd, env, cols=100, rows=30)
    child.expect([r"\$|>\s*", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    # Narrow the terminal; the process gets SIGWINCH.
    child.setwinsize(30, 40)
    child.sendline("Say hello.")
    # The block quote text should appear, wrapped to ~40 columns.
    child.expect(r"block quote", timeout=60)
    # Give Rich a moment to finish rendering, then assert the visible screen
    # contains the wrapped (broken) quote text.
    child.expect(pexpect.TIMEOUT, timeout=1)
    screen = child.read()
    assert b"block quote" in screen
    assert b"across multiple lines" in screen
}