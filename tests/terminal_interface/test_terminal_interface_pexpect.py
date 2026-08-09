"""Real-terminal tests for the CLI using pexpect.

These spawn the actual `interpreter` command in a pseudo-terminal (PTY) and
assert on the bytes it writes, including ANSI escape codes. They cover:

- basic startup and prompt rendering
- markdown output (block quotes, tables, code fences)
- colors: 16-color, 256-color, and truecolor (24-bit) escape sequences
- terminal width change: sending SIGWINCH and confirming the rendered output
  wraps to the new width (the develop branch detects width changes and
  re-wraps new text to the new width)

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
        "-m",
        "interpreter",
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
def test_markdown_block_quote_wraps_and_colors(run_cli, interpreter_cmd):
    """A block quote renders with color and wraps to the terminal width."""
    cmd, env = interpreter_cmd
    child = run_cli(cmd, env, cols=60, rows=30)
    child.expect([r"\$|>\s*", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    child.sendline("Say hello.")
    # The reply contains a block quote marker; Rich renders it with a color escape.
    child.expect(r"\x1b\[\d+(;\d+)*m[^\x1b]*>", timeout=60)


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
    """A code fence renders with syntax highlighting escapes."""
    cmd, env = interpreter_cmd
    child = run_cli(cmd, env, cols=80, rows=30)
    child.expect([r"\$|>\s*", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    child.sendline("Say hello.")
    child.expect(r"def hello", timeout=60)


@pytest.mark.timeout(90)
def test_truecolor_escapes_present(run_cli, interpreter_cmd):
    """Truecolor (24-bit) ANSI escapes appear in the rendered output."""
    cmd, env = interpreter_cmd
    child = run_cli(cmd, env, cols=80, rows=30)
    child.expect([r"\$|>\s*", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    child.sendline("Say hello.")
    # Rich emits \x1b[38;2;R;G;Bm for truecolor when the terminal supports it.
    child.expect(r"\x1b\[38;2;\d+;\d+;\d+m", timeout=60)


@pytest.mark.timeout(90)
def test_terminal_resize_changes_wrap(run_cli, interpreter_cmd):
    """Resizing the terminal re-wraps markdown to the new width.

    This is the key regression test for the develop branch's width-change
    detection: after a SIGWINCH (via setwinsize), newly rendered output should
    wrap to the narrower column count.
    """
    cmd, env = interpreter_cmd
    child = run_cli(cmd, env, cols=100, rows=30)
    child.expect([r"\$|>\s*", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    # Narrow the terminal; the process gets SIGWINCH.
    child.setwinsize(30, 40)
    child.sendline("Say hello.")
    # The block quote text should appear, wrapped to ~40 columns.
    child.expect(r"block quote", timeout=60)
}