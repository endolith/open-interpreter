from interpreter.terminal_interface.render_past_conversation import (
    render_past_conversation,
)


def _code_output_messages(output):
    return [
        {
            "role": "assistant",
            "type": "code",
            "format": "bash",
            "content": "ls",
        },
        {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": output,
        },
    ]


def test_replay_strips_mouse_tracking_enables(capsys):
    """Replaying a TUI-poisoned log must not re-emit tracking enables."""
    output = "screen redraw\x1b[?1000h\x1b[?1006h\x1b[?1049h done"
    render_past_conversation(_code_output_messages(output))
    out = capsys.readouterr().out
    assert "?1000h" not in out
    assert "?1006h" not in out
    assert "?1049h" not in out
    assert "screen redraw" in out


def test_replay_strips_device_queries(capsys):
    """Replaying must not re-emit queries whose replies would land in input."""
    output = "probe\x1b[6n\x1b[14t done"
    render_past_conversation(_code_output_messages(output))
    out = capsys.readouterr().out
    assert "[6n" not in out
    assert "[14t" not in out
    assert "probe" in out


def test_replay_keeps_plain_colors(capsys):
    """The replay strip is surgical — ordinary SGR colors still display."""
    output = "\x1b[32mgreen\x1b[0m"
    render_past_conversation(_code_output_messages(output))
    out = capsys.readouterr().out
    assert "green" in out
    assert "[32m" in out
