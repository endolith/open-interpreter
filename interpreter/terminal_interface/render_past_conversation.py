"""
This is all messed up.... Uses the old streaming structure.
"""

from rich import print as rich_print
from rich.markdown import Markdown
from .utils.display_markdown_message import display_markdown_message


def _render_code_block(code, output, language):
    language = language or "text"
    rich_print(Markdown(f"```{language}\n{code or ''}\n```"))
    if (output or "").strip():
        rich_print(Markdown(f"```text\n{output.strip()}\n```"))


def render_past_conversation(messages):
    # History replay should not use incremental/live block rendering.
    # Messages in saved conversations are already complete.
    pending_code = None
    pending_output = ""

    def flush_pending_code():
        nonlocal pending_code, pending_output
        if pending_code is None:
            return
        _render_code_block(
            pending_code.get("content", ""),
            pending_output,
            pending_code.get("format"),
        )
        pending_code = None
        pending_output = ""

    for chunk in messages:
        role = chunk.get("role")
        chunk_type = chunk.get("type")
        content = chunk.get("content", "")

        if role == "user":
            flush_pending_code()
            if isinstance(content, str):
                print(">", content)
            continue

        if role == "assistant" and chunk_type == "message":
            flush_pending_code()
            if isinstance(content, str) and content.strip():
                display_markdown_message(content)
            continue

        if role == "assistant" and chunk_type == "code":
            flush_pending_code()
            pending_code = chunk
            pending_output = ""
            continue

        if role == "computer" and chunk_type == "console":
            if chunk.get("format") == "active_line":
                continue
            if isinstance(content, str):
                pending_output += "\n" + content
            continue

    flush_pending_code()
