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
