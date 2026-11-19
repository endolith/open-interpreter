"""Buffered shell output must bracket its non-command metadata.

The user-facing output panel renders the note, exit state, and elapsed time in
the same block as real command output. Brackets are the only visual cue that
those lines describe the run rather than come from the program, so they must
survive changes to the buffered-output path.
"""

from types import SimpleNamespace

from interpreter.core.terminal.terminal import Terminal


class _FakeBash:
    """Minimal language double that emits command output plus a state line."""

    def __init__(self, state_line=True):
        self.state_line = state_line

    def run(self, code):
        yield {"type": "console", "format": "output", "content": "hello world"}
        if self.state_line:
            yield {
                "type": "console",
                "format": "output",
                "content": "[Exit code: 0 | Shell state: CWD: /tmp]",
                "state_line": True,
            }


def _output_content(terminal, language="bash"):
    """Collect the output chunks into the string the panel will render."""
    chunks = list(terminal._streaming_run(language, "echo hello"))
    return "\n".join(
        chunk["content"]
        for chunk in chunks
        if chunk.get("type") == "console" and chunk.get("format") == "output"
    )


def test_buffered_output_metadata_is_bracketed():
    """The note, exit state, and elapsed time are marked as metadata, not stdout."""
    interpreter = SimpleNamespace(highlight_active_line=True)
    terminal = Terminal(interpreter)
    terminal._active_languages["bash"] = _FakeBash()

    content = _output_content(terminal)

    assert content.startswith(
        "[Note: Shell command output will be shown after completion.]"
    )
    assert "hello world" in content
    assert "[Exit code: 0 | Shell state: CWD: /tmp | Elapsed: " in content
    assert "\n\n[Elapsed:" not in content


def test_buffered_output_combines_state_and_elapsed_in_one_bracket():
    """Elapsed time joins the exit-state bracket instead of a second line."""
    interpreter = SimpleNamespace(highlight_active_line=True)
    terminal = Terminal(interpreter)
    terminal._active_languages["bash"] = _FakeBash()

    content = _output_content(terminal)

    assert "\nhello world\n[Exit code: 0 | Shell state: CWD: /tmp | Elapsed: " in content


def test_buffered_output_without_state_line_has_one_elapsed_bracket():
    """Languages that emit no state line still get a bracketed elapsed time."""
    interpreter = SimpleNamespace(highlight_active_line=True)
    terminal = Terminal(interpreter)
    terminal._active_languages["bash"] = _FakeBash(state_line=False)

    content = _output_content(terminal)

    assert "\nhello world\n[Elapsed: " in content
    assert "[Exit code" not in content
