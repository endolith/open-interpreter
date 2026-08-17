def export_to_markdown(messages: list[dict], export_path: str, include_reasoning=True):
    markdown = messages_to_markdown(messages, include_reasoning=include_reasoning)
    with open(export_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    print(f"Exported current conversation to {export_path}")


def messages_to_markdown(messages: list[dict], include_reasoning=True) -> str:
    # Convert interpreter.messages to Markdown text
    markdown_content = ""
    previous_role = None
    for chunk in messages:
        # Skip reasoning blocks entirely when they aren't wanted, before the
        # role-header logic runs, so a skipped reasoning message doesn't steal
        # the "## assistant" header from the actual response that follows it.
        if not include_reasoning and chunk.get("format") == "reasoning":
            continue

        current_role = chunk["role"]
        current_role = chunk["role"]
        if current_role == previous_role:
            rendered_chunk = ""
        else:
            rendered_chunk = f"## {current_role}\n\n"
            previous_role = current_role

        # User query message
        if chunk["role"] == "user":
            rendered_chunk += chunk["content"] + "\n\n"
            markdown_content += rendered_chunk
            continue

        # Message
        if chunk["type"] == "message":
            if chunk.get("format") == "reasoning":
                rendered_chunk += _reasoning_to_blockquote(chunk["content"])
            else:
                rendered_chunk += chunk["content"] + "\n\n"

        # Code
        if chunk["type"] == "code" or chunk["type"] == "console":
            code_format = chunk.get("format", "")
            rendered_chunk += f"```{code_format}\n{chunk['content']}\n```\n\n"

        markdown_content += rendered_chunk

    return markdown_content


def _reasoning_to_blockquote(content: str) -> str:
    # Prefix every line with "> " so the model's thoughts render as a blockquote.
    # The trailing newlines that the reasoning chunks are yielded with (e.g.
    # "\n\n") are handled by the loop's own separator, so strip them here and
    # re-add exactly one to keep the markdown tidy.
    lines = content.rstrip("\n").split("\n")
    quoted = "\n".join(f"> {line}" for line in lines)
    return quoted + "\n\n"
