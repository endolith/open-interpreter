"""Signal for a tool call that `run_tool_calling_llm` could not turn into code.

The producer can receive a call it cannot run — an unknown function name, or
`execute` without both `language` and `code`. Nothing is appended to the
transcript in that case, so the turn used to end silently (#629). It now
stores the problem as assistant text so the model sees the failure, and
`respond` gives a `--no-loop` session one retry turn when the report is
marked as shown below.

The retry decision cannot look for the error text in the transcript: an
ordinary answer that quotes it would then look like a failure, and
`_respond_and_store` merges narration and report chunks into one message, so
the stored entry does not reliably identify the report either. The producer
therefore marks the report chunk itself with `UNRUNNABLE_TOOL_CALL_KEY`, and
`respond` records that signal while streaming the current call and strips the
key before the chunk is stored.
"""

TOOL_CALL_ERROR_PREFIX = "**Error:** Tool call"

# Set only on the chunk `run_tool_calling_llm` emits for a call it cannot run.
UNRUNNABLE_TOOL_CALL_KEY = "unrunnable_tool_call"


def is_unrunnable_tool_call_chunk(chunk):
    """Whether a stream chunk is the producer's explicit unrunnable-call signal."""
    return chunk.get(UNRUNNABLE_TOOL_CALL_KEY) is True


def without_unrunnable_tool_call_key(chunk):
    """Return the chunk with the signal key removed for storage and consumers."""
    return {
        key: value
        for key, value in chunk.items()
        if key != UNRUNNABLE_TOOL_CALL_KEY
    }
