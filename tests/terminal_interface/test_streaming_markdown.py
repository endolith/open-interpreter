from interpreter.terminal_interface.utils.streaming_markdown import (
    detect_complete_block,
    textify_markdown_code_blocks,
)

LIST_WITH_INDENTED_FENCE = """1. **First**

2. **Second**

3. **Third**

4. **Run PowerShell**:
   ```powershell
   Get-Service | Set-Service -StartupType Disabled
   ```

5. **Fifth**

6. **Sixth**

7. **Seventh**
"""


def test_textify_preserves_indented_fence():
    textified = textify_markdown_code_blocks(LIST_WITH_INDENTED_FENCE)
    assert "   ```text" in textified
    assert "\n```powershell" not in textified


def test_detect_complete_block_keeps_list_with_code_fence_together():
    textified = textify_markdown_code_blocks(LIST_WITH_INDENTED_FENCE)
    result = detect_complete_block(textified)
    assert result is None


def test_detect_complete_block_defers_standalone_hr():
    hr_and_list = """---

1. Level 1
   - Level 2
"""
    assert detect_complete_block(hr_and_list) is None


def test_detect_complete_block_commits_hr_with_following_block():
    text = """Intro paragraph.

---

1. First

2. Second

---

1. Level 1
   - Level 2
"""
    first = detect_complete_block(text)
    assert first is not None
    block, next_line = first
    assert "Intro paragraph" in block
    assert "---" not in block

    remaining = "\n".join(text.split("\n")[next_line:])
    second = detect_complete_block(remaining)
    assert second is not None
    block2, next_line2 = second
    assert "First" in block2 and "Second" in block2
    assert block2.startswith("---")
    assert "Level 1" not in block2

    remaining2 = "\n".join(remaining.split("\n")[next_line2:])
    third = detect_complete_block(remaining2)
    assert third is None


def test_refresh_live_display_updates_in_place():
    import os

    os.environ["TERM"] = "xterm-256color"
    from rich.text import Text

    from interpreter.terminal_interface.utils.streaming_markdown import (
        create_live_display,
        refresh_live_display,
    )
    from tests.terminal_interface.virtual_terminal import make_tty_console

    console, vt, _ = make_tty_console(80, 12)
    live = create_live_display(console)
    live.start()
    refresh_live_display(live, Text("first\nsecond"))
    assert live._live_render._shape is not None
    refresh_live_display(live, Text("third"))
    assert live._live_render._shape is not None
    live.stop()


def test_clear_live_shape_before_print():
    import os

    os.environ["TERM"] = "xterm-256color"
    from rich.text import Text

    from interpreter.terminal_interface.utils.streaming_markdown import (
        clear_live_shape,
        create_live_display,
        refresh_live_display,
    )
    from tests.terminal_interface.virtual_terminal import make_tty_console

    console, vt, _ = make_tty_console(80, 12)
    live = create_live_display(console)
    live.start()
    refresh_live_display(live, Text("tall\nline\ntwo\nthree"))
    assert live._live_render._shape is not None
    clear_live_shape(live)
    assert live._live_render._shape is None
    live.stop()


def test_stop_live_display_clear_false_skips_empty_update():
    import os

    os.environ["TERM"] = "xterm-256color"
    from rich.text import Text

    from interpreter.terminal_interface.utils.streaming_markdown import (
        create_live_display,
        refresh_live_display,
        stop_live_display,
    )
    from tests.terminal_interface.virtual_terminal import make_tty_console

    console, _, buf = make_tty_console(80, 12)
    live = create_live_display(console)
    live.start()
    refresh_live_display(live, Text("preview line"))
    stop_live_display(live, clear=False)
    assert not live.is_started
    assert "preview line" in buf.raw


def test_stop_live_display_noop_when_not_started():
    from interpreter.terminal_interface.utils.streaming_markdown import (
        create_live_display,
        stop_live_display,
    )
    from tests.terminal_interface.virtual_terminal import make_tty_console

    console, _, _ = make_tty_console(80, 12)
    live = create_live_display(console)
    stop_live_display(live)


def test_create_sliding_window_display_truncates_with_ellipsis():
    import io

    from rich.console import Console

    from interpreter.terminal_interface.utils.streaming_markdown import (
        create_sliding_window_display,
    )
    from tests.terminal_interface.virtual_terminal import make_tty_console

    console, _, _ = make_tty_console(80, 8)
    lines = [f"line {i}" for i in range(10)]
    display = create_sliding_window_display(console, lines, viewport_lines=3)
    buf = io.StringIO()
    Console(file=buf, width=80, height=8, force_terminal=True).print(display)
    rendered = buf.getvalue()
    assert "..." in rendered
    assert "line 9" in rendered
    assert "line 0" not in rendered


def test_calculate_window_size_respects_fraction():
    from interpreter.terminal_interface.utils.streaming_markdown import (
        calculate_window_size,
    )
    from tests.terminal_interface.virtual_terminal import make_tty_console

    console, _, _ = make_tty_console(80, 20)
    assert calculate_window_size(console, 0.5) == 10
    assert calculate_window_size(console, 0.0) == 1
