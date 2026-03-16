"""
This is all messed up.... Uses the old streaming structure.
"""

from rich.console import Group
from rich import print as rich_print
from rich.box import ROUNDED
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from .utils.display_constants import PADDING_PANEL
from .utils.display_markdown_message import display_markdown_message


def _render_code_block(code, output, language):
    language = language or "text"
    code_md = Markdown(f"```{language}\n{code or ''}\n```")
    if (output or "").strip():
        output_md = Markdown(f"```text\n{output.strip()}\n```")
        rich_print(Padding(Group(code_md, output_md), PADDING_PANEL))
    else:
        rich_print(Padding(code_md, PADDING_PANEL))


def render_past_conversation(messages):
    # History replay should not use incremental/live block rendering.
    # Messages in saved conversations are already complete.
    pending_code = None
    pending_output = ""
    has_rendered_message = False

    def render_separator():
        nonlocal has_rendered_message
        if has_rendered_message:
            print("")
        has_rendered_message = True

    def flush_pending_code():
        nonlocal pending_code, pending_output
        if pending_code is None:
            return
        render_separator()
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

        if chunk_type == "view_image_call":
            flush_pending_code()
            path = chunk.get("path", "")
            if path:
                render_separator()
                panel = Panel(
                    f"AI requested to view image:\n{path}",
                    box=ROUNDED,
                    title="View Image Request",
                    padding=(0, 1),
                )
                rich_print(Padding(panel, PADDING_PANEL))
            continue

        if role == "user":
            flush_pending_code()
            # Skip UI-injected messages (e.g. [User declined...], [The user edited...])
            # so they don't appear as user speech when loading from history.
            if chunk.get("source") == "terminal":
                continue
            if chunk_type == "image" and isinstance(content, str):
                render_separator()
                panel = Panel(
                    f"User approved. Image shown to model:\n{content}",
                    box=ROUNDED,
                    title="Image Approved",
                    padding=(0, 1),
                )
                rich_print(Padding(panel, PADDING_PANEL))
                continue
            if isinstance(content, str):
                render_separator()
                print(">", content)
            continue

        if role == "assistant" and chunk_type == "message":
            flush_pending_code()
            if isinstance(content, str) and content.strip():
                render_separator()
                if chunk.get("format") == "reasoning":
                    markdown = Markdown(content.strip(), style="cyan")
                    panel = Panel(markdown, box=ROUNDED, border_style="cyan", title="Thinking")
                    rich_print(Padding(panel, PADDING_PANEL))
                else:
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
