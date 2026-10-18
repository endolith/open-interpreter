FENCE = "```"


def _withheld_fence_chars(text):
    """How many trailing backticks to hold back because they may still grow into a fence.

    A stream can split ``` across chunks, so a trailing run of one or two
    backticks is ambiguous: it may be literal prose (`ls`) or the start of a
    fence. Those are withheld until the next chunk disambiguates them. A run of
    three is already a fence, so nothing is withheld.
    """
    # Check for a complete fence before counting, rather than counting and
    # comparing the count to len(FENCE) afterward. Order matters: a run of four
    # also ends with three, and "``" is itself a suffix of the fence, so
    # counting first makes it easy to read the comparison as redundant.
    if text.endswith(FENCE):
        return 0
    return len(text) - len(text.rstrip("`"))


def _find_closing_fence(text, start, at_eof=False):
    """Find a CommonMark closing fence at or after ``start``.

    A closing fence is a line with up to three leading spaces, three or more
    backticks, and only spaces/tabs after them. Mid-line fences and fences with
    trailing text are code content.

    Returns ``(fence_start, safe_end)``. ``fence_start`` is None when no
    complete fence is present. ``safe_end`` is the point up to which code can
    be emitted now: it holds an unterminated line that could still grow into a
    fence, and otherwise extends to the end of the buffer.
    """
    pos = start
    while pos <= len(text):
        newline = text.find("\n", pos)
        end = len(text) if newline == -1 else newline
        line = text[pos:end]
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        backticks = len(stripped) - len(stripped.lstrip("`"))
        trailing = stripped[backticks:]
        if indent <= 3 and backticks >= len(FENCE) and not trailing.strip():
            if newline != -1 or at_eof:
                return pos, pos
            return None, pos
        if newline == -1:
            # An unterminated tail made only of spaces/backticks can still grow
            # into a fence; hold it. Anything else is ordinary code.
            if (
                indent <= 3
                and not trailing.strip()
                and set(stripped) <= {"`", " ", "\t"}
            ):
                return None, pos
            return None, len(text)
        pos = newline + 1
    return None, len(text)


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

    # Everything received so far, and how much of it has already been yielded.
    # Tracking a cursor into one buffer (rather than emitting each chunk as it
    # arrives) is what lets a chunk be held back for fence disambiguation and
    # still be emitted afterwards, instead of being dropped.
    accumulated = ""
    cursor = 0

    inside_code_block = False
    language = None
    # Where the opening fence began, so that a stream cut off before the
    # language line completes can still emit the fence and the partial label it
    # already consumed, rather than dropping them.
    block_open_at = 0

    for chunk in llm.completions(**params):
        if llm.interpreter.verbose:
            print("Chunk in coding_llm", chunk)

        if "choices" not in chunk or len(chunk["choices"]) == 0:
            # This happens sometimes
            continue

        content = chunk["choices"][0]["delta"].get("content", "")

        if content == None:
            continue

        accumulated += content

        # One chunk can cross a boundary — prose, then a fence, then code — so
        # keep consuming the buffer until it needs more input.
        while True:
            if not inside_code_block:
                fence_at = accumulated.find(FENCE, cursor)

                if fence_at == -1:
                    # No fence yet: emit the prose that cannot be part of one.
                    safe_end = len(accumulated) - _withheld_fence_chars(accumulated)
                    if safe_end > cursor:
                        yield {
                            "type": "message",
                            "content": accumulated[cursor:safe_end],
                        }
                        cursor = safe_end
                    break

                # Emit any prose preceding the fence, then enter the block.
                if fence_at > cursor:
                    yield {
                        "type": "message",
                        "content": accumulated[cursor:fence_at],
                    }
                block_open_at = fence_at
                cursor = fence_at + len(FENCE)
                inside_code_block = True
                continue

            # Inside a code block. The language label occupies the rest of the
            # line after the opening fence and is structural, not code, so it is
            # consumed by advancing the cursor past it rather than by stripping
            # the label out of the emitted text.
            if language is None:
                newline_at = accumulated.find("\n", cursor)
                if newline_at == -1:
                    break  # the language line is still arriving

                # Assigned before being refined so that an unlabelled fence in
                # OS mode settles on "" rather than staying None, which would
                # make the next line be parsed as a label.
                language = accumulated[cursor:newline_at]

                # Default to python if not specified
                if language == "":
                    if llm.interpreter.os == False:
                        language = "python"
                    elif llm.interpreter.os == False:
                        # OS mode does this frequently. Takes notes with markdown code blocks
                        # NOTE: repeats the condition above, so it is unreachable
                        # and OS mode emits nothing here — see issue #220.
                        language = "text"
                else:
                    # Removes hallucinations containing spaces or non letters.
                    language = "".join(char for char in language if char.isalpha())
                cursor = newline_at + 1

            fence_at, safe_end = _find_closing_fence(accumulated, cursor)

            if fence_at is None:
                if safe_end > cursor:
                    if language:
                        yield {
                            "type": "code",
                            "format": language,
                            "content": accumulated[cursor:safe_end],
                        }
                    cursor = safe_end
                break

            # Closing fence: emit the remaining body and stop.
            if fence_at > cursor and language:
                yield {
                    "type": "code",
                    "format": language,
                    "content": accumulated[cursor:fence_at],
                }
            return

    # A closing fence may be complete only because the stream ended (no
    # trailing newline); resolve it before flushing the tail as code.
    if inside_code_block and language is not None:
        fence_at, _ = _find_closing_fence(accumulated, cursor, at_eof=True)
        if fence_at is not None:
            if fence_at > cursor:
                yield {
                    "type": "code",
                    "format": language,
                    "content": accumulated[cursor:fence_at],
                }
            return

    # The stream ended with something still unemitted.
    if inside_code_block and language is None:
        # The fence opened but its language line never finished, so this was
        # never confirmed as a code block. Emit the fence and whatever label
        # text arrived as literal message content rather than swallowing it —
        # the whole point of this function is to never silently drop what the
        # model sent.
        yield {"type": "message", "content": accumulated[block_open_at:]}
    # The stream ended with backticks still held back, so they were literal
    # text rather than the start of a fence. Emit them instead of dropping them.
    elif cursor < len(accumulated):
        if not inside_code_block:
            yield {"type": "message", "content": accumulated[cursor:]}
        elif language:
            yield {"type": "code", "format": language, "content": accumulated[cursor:]}
