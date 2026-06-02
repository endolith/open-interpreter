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
