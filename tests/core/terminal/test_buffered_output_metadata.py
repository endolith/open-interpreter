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

    def run(self, code):
        yield {"type": "console", "format": "output", "content": "hello world"}
        yield {
            "type": "console",
            "format": "output",
            "content": "[Exit code: 0 | Shell State: CWD: /tmp]",
        }


def test_buffered_output_metadata_is_bracketed():
    """The buffering note and elapsed time are marked as metadata, not stdout."""
    interpreter = SimpleNamespace(highlight_active_line=True)
    terminal = Terminal(interpreter)
    terminal._active_languages["bash"] = _FakeBash()

    chunks = list(terminal._streaming_run("bash", "echo hello"))
    content = "\n".join(
        chunk["content"]
        for chunk in chunks
        if chunk.get("type") == "console" and chunk.get("format") == "output"
    )

    assert content.startswith(
        "[Note: Shell command output will be shown after completion.]"
    )
    assert "hello world" in content
    assert "[Exit code: 0 | Shell State: CWD: /tmp]" in content
    assert "[Time elapsed:" in content
    assert "\n\nTime elapsed:" not in content
