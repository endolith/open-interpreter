"""
This is all messed up.... Uses the old streaming structure.
"""


from .components.code_block import CodeBlock
from .components.message_block import MessageBlock
from .utils.display_markdown_message import display_markdown_message


def render_past_conversation(messages):
    # This is a clone of the terminal interface.
    # So we should probably find a way to deduplicate...

    active_block = None
    render_cursor = False
    ran_code_block = False

    for chunk in messages:
        # Only addition to the terminal interface:
        if chunk.get("role") == "user":
            if active_block:
                if getattr(active_block, "type", None) == "message" and hasattr(
                    active_block, "finalize"
                ):
                    active_block.finalize()
                active_block.end()
                active_block = None
            content = chunk.get("content", "")
            if isinstance(content, str):
                print(">", content)
            continue

        # Message (assistant). MessageBlock displays from .buffer, not .message.
        if chunk.get("type") == "message":
            if active_block is None:
                active_block = MessageBlock()
            if active_block.type != "message":
                active_block.end()
                active_block = MessageBlock()
            content = chunk.get("content", "")
            if isinstance(content, str):
                active_block.buffer += content

        # Code
        if chunk.get("type") == "code":
            if active_block is None:
                active_block = CodeBlock()
            if active_block.type != "code" or ran_code_block:
                # If the last block wasn't a code block,
                # or it was, but we already ran it:
                if getattr(active_block, "type", None) == "message" and hasattr(
                    active_block, "finalize"
                ):
                    active_block.finalize()
                active_block.end()
                active_block = CodeBlock()
            ran_code_block = False
            render_cursor = True

            if "format" in chunk:
                active_block.language = chunk["format"]
            content = chunk.get("content", "")
            if isinstance(content, str):
                active_block.code += content
            if "active_line" in chunk:
                active_block.active_line = chunk["active_line"]

        # Console
        if chunk.get("type") == "console":
            ran_code_block = True
            render_cursor = False
            # Console output should be associated with a code block
            if active_block is None:
                active_block = CodeBlock()
            if active_block.type != "code":
                if getattr(active_block, "type", None) == "message" and hasattr(
                    active_block, "finalize"
                ):
                    active_block.finalize()
                active_block.end()
                active_block = CodeBlock()
            content = chunk.get("content", "")
            if isinstance(content, str):
                active_block.output += "\n" + content
            active_block.output = active_block.output.strip()  # <- Aesthetic choice

        if active_block:
            active_block.refresh(cursor=render_cursor)

    # (Sometimes -- like if they CTRL-C quickly -- active_block is still None here)
    if active_block:
        if getattr(active_block, "type", None) == "message" and hasattr(
            active_block, "finalize"
        ):
            active_block.finalize()
        active_block.end()
        active_block = None
