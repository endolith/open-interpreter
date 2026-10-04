def run_text_llm(llm, params):
    """Stream an LLM's response as typed chunks, parsing code fences.

    Takes an LLM wrapper and a params dict (passed to ``llm.completions``).
    Yields chunks of two types:

    - ``{"type": "message", "content": <str>}`` – plain text from the stream.
    - ``{"type": "code", "format": <lang>, "content": <str>}`` – content from
      inside fenced code blocks (`` ```<lang>``).

    The first code delta strips the leading language line so language names do
    not appear inside the yielded code body.  Subsequent deltas for the same
    block are passed through unchanged, preventing repeated stripping from
    corrupting code that contains the language name.

    Args:
        llm: Object with a ``completions`` method (yields API response dicts)
             and an ``execution_instructions`` attribute (optional string prepended
             to the system prompt).
        params: Dict passed to ``llm.completions(**params)``; must contain a
                ``messages`` list whose first element has a ``content`` field.

    Yields:
        Chunks as described above.
    """
    ## Setup

    if llm.execution_instructions:
        try:
            # Add the system message
            params["messages"][0][
                "content"
            ] += "\n" + llm.execution_instructions
        except:
            print('params["messages"][0]', params["messages"][0])
            raise

    ## Convert output to LMC format

    inside_code_block = False
    accumulated_block = ""
    language = None
    # Tracks whether we've already yielded the first code delta (with language stripped)
    language_line_yielded = False

    for chunk in llm.completions(**params):
        if llm.interpreter.verbose:
            print("Chunk in coding_llm", chunk)

        if "choices" not in chunk or len(chunk["choices"]) == 0:
            # This happens sometimes
            continue

        content = chunk["choices"][0]["delta"].get("content", "")

        if content == None:
            continue

        accumulated_block += content

        if accumulated_block.endswith("`"):
            # We might be writing "```" one token at a time.
            continue

        # Did we just enter a code block?
        if "```" in accumulated_block and not inside_code_block:
            inside_code_block = True
            accumulated_block = accumulated_block.split("```")[1]

        # Did we just exit a code block?
        if inside_code_block and "```" in accumulated_block:
            return

        # If we're in a code block,
        if inside_code_block:
            # If we don't have a `language`, find it
            if language is None and "\n" in accumulated_block:
                language = accumulated_block.split("\n")[0]

                # Default to python if not specified
                if language == "":
                    if llm.interpreter.os == False:
                        language = "python"
                    elif llm.interpreter.os == False:
                        # OS mode does this frequently. Takes notes with markdown code blocks
                        language = "text"
                else:
                    # Removes hallucinations containing spaces or non letters.
                    language = "".join(char for char in language if char.isalpha())

            # If we do have a `language`, send it out
            if language:
                if not language_line_yielded:
                    # First code delta: strip the language line from accumulated_block
                    # and yield the remainder as the initial code content.
                    language_line_yielded = True
                    code_content = accumulated_block
                    # Remove the leading language line (everything before the first \n)
                    if "\n" in code_content:
                        code_content = code_content.split("\n", 1)[1]
                    yield {
                        "type": "code",
                        "format": language,
                        "content": code_content,
                    }
                else:
                    # Subsequent deltas: yield content unchanged.
                    yield {
                        "type": "code",
                        "format": language,
                        "content": content,
                    }

        # If we're not in a code block, send the output as a message
        if not inside_code_block:
            yield {"type": "message", "content": content}
