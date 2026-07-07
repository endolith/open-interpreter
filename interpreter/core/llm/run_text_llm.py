FENCE = "```"


def _reasoning_chunk(reasoning):
    """A reasoning-only message chunk so the thought reaches stored history."""
    return {"type": "message", "content": "", "reasoning_content": reasoning}


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


def _find_closing_fence(text, start, at_eof=False, opener_len=len(FENCE)):
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
    # ``start`` can be in the middle of a code line: the caller emits a partial
    # line and resumes there when the next delta arrives. A closing fence must
    # begin at a physical line start, so a run of backticks at ``start`` with
    # text before it is code (e.g. ``# ``` ``), not a closer. Skip the rest of
    # that line before scanning for one.
    if pos > 0 and text[pos - 1] != "\n":
        newline = text.find("\n", pos)
        if newline == -1:
            return None, len(text)
        pos = newline + 1
    while pos < len(text):
        newline = text.find("\n", pos)
        end = len(text) if newline == -1 else newline
        line = text[pos:end]
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        backticks = len(stripped) - len(stripped.lstrip("`"))
        trailing = stripped[backticks:]
        if indent <= 3 and backticks >= opener_len and not trailing.strip():
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
    skip_execution_instructions = params.pop("skip_execution_instructions", False)
    plain_title_stream = params.pop("conversation_title_plain_stream", False)

    ## Setup

    if llm.execution_instructions and not skip_execution_instructions:
        try:
            # Add the system message
            params["messages"][0]["content"] += "\n" + llm.execution_instructions
        except:
            print('params["messages"][0]', params["messages"][0])
            raise

    if plain_title_stream:
        for chunk in llm.completions(**params):
            if llm.interpreter.verbose:
                print("Chunk in coding_llm", chunk)

            if "choices" not in chunk or len(chunk["choices"]) == 0:
                continue
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content", "")
            if content is None or content == "":
                continue
            yield {"type": "message", "content": content}

        return

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
    # The opening run can be longer than FENCE (e.g. ```` for Markdown that
    # contains a bare ``` line). The closing run must be at least as long.
    opener_len = len(FENCE)

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
        if content is None and not reasoning:
            continue
        content = content or ""
        pending_reasoning = reasoning

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
                        message = {
                            "type": "message",
                            "content": accumulated[cursor:safe_end],
                        }
                        if pending_reasoning:
                            message["reasoning_content"] = pending_reasoning
                            pending_reasoning = ""
                        yield message
                        cursor = safe_end
                    break

                # Emit any prose preceding the fence, then enter the block.
                if fence_at > cursor:
                    message = {
                        "type": "message",
                        "content": accumulated[cursor:fence_at],
                    }
                    if pending_reasoning:
                        message["reasoning_content"] = pending_reasoning
                        pending_reasoning = ""
                    yield message
                block_open_at = fence_at
                opener_end = fence_at
                while opener_end < len(accumulated) and accumulated[opener_end] == "`":
                    opener_end += 1
                opener_len = opener_end - fence_at
                cursor = opener_end
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

                # A run split across deltas leaves extra backticks at the start
                # of the info line. Count them toward the opener so a shorter
                # inner run cannot close the block.
                label = accumulated[cursor:newline_at]
                extra = len(label) - len(label.lstrip("`"))
                if extra:
                    opener_len += extra
                    cursor += extra
                    label = label[extra:]

                # Assigned before being refined so that an unlabelled fence in
                # OS mode settles on "" rather than staying None, which would
                # make the next line be parsed as a label.
                language = label.strip()

                # Default to python if not specified
                if language == "":
                    if llm.interpreter.os == False:
                        language = "python"
                    elif llm.interpreter.os == False:  # pragma: no cover - see #392
                        # OS mode does this frequently. Takes notes with markdown code blocks
                        # NOTE: repeats the condition above, so it is unreachable
                        # and OS mode emits nothing here — see issue #220.
                        language = "text"
                else:
                    # Removes hallucinations containing spaces or non letters.
                    language = "".join(char for char in language if char.isalpha())
                cursor = newline_at + 1

            fence_at, safe_end = _find_closing_fence(
                accumulated, cursor, opener_len=opener_len
            )

            if fence_at is None:
                if safe_end > cursor:
                    if pending_reasoning:
                        yield _reasoning_chunk(pending_reasoning)
                        pending_reasoning = ""
                    if language:
                        yield {
                            "type": "code",
                            "format": language,
                            "content": accumulated[cursor:safe_end],
                        }
                    cursor = safe_end
                break

            # Closing fence: emit the remaining body and stop.
            if pending_reasoning:
                yield _reasoning_chunk(pending_reasoning)
                pending_reasoning = ""
            if fence_at > cursor and language:
                yield {
                    "type": "code",
                    "format": language,
                    "content": accumulated[cursor:fence_at],
                }
            return

        if pending_reasoning:
            yield _reasoning_chunk(pending_reasoning)

    # A closing fence may be complete only because the stream ended (no
    # trailing newline); resolve it before flushing the tail as code.
    if inside_code_block and language is not None:
        fence_at, _ = _find_closing_fence(
            accumulated, cursor, at_eof=True, opener_len=opener_len
        )
        if fence_at is not None:
            if fence_at > cursor:  # pragma: no cover - content already emitted
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
