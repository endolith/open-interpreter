def run_text_llm(llm, params):
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

    for chunk in llm.completions(**params):
        if llm.interpreter.verbose:
            print("Chunk in coding_llm", chunk)

        if "choices" not in chunk or len(chunk["choices"]) == 0:
            # This happens sometimes
            continue

        delta = chunk["choices"][0]["delta"]
        content = delta.get("content", "")
        # Thinking models (e.g. DeepSeek via OpenRouter) stream their chain
        # of thought separately; providers require it passed back verbatim.
        reasoning = delta.get("reasoning_content") or ""

        if content == None and not reasoning:
            continue

        content = content or ""

        accumulated_block += content

        if accumulated_block.endswith("`"):
            # We might be writing "```" one token at a time.
            if reasoning:
                # The fence parser suppresses this chunk; emit thinking-model
                # reasoning on its own so providers get it back verbatim.
                yield {
                    "type": "message",
                    "content": "",
                    "reasoning_content": reasoning,
                }
            continue

        # Did we just enter a code block?
        if "```" in accumulated_block and not inside_code_block:
            inside_code_block = True
            accumulated_block = accumulated_block.split("```")[1]

        # Did we just exit a code block?
        if inside_code_block and "```" in accumulated_block:
            if reasoning:
                # The closing fence ends the turn; emit reasoning first so it
                # still reaches stored history.
                yield {
                    "type": "message",
                    "content": "",
                    "reasoning_content": reasoning,
                }
            return

        # If we're in a code block,
        if inside_code_block:
            if reasoning:
                # Code chunks carry no reasoning; emit it on its own message
                # so it still reaches stored history.
                yield {
                    "type": "message",
                    "content": "",
                    "reasoning_content": reasoning,
                }
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
                yield {
                    "type": "code",
                    "format": language,
                    "content": content.replace(language, ""),
                }

        # If we're not in a code block, send the output as a message
        if not inside_code_block:
            message_chunk = {"type": "message", "content": content}
            if reasoning:
                message_chunk["reasoning_content"] = reasoning
            yield message_chunk
