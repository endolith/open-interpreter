"""Marker for the assistant message that reports a tool call we could not run.

`run_tool_calling_llm` can receive a tool call it cannot turn into code —
an unknown function name, or `execute` without both `language` and
`code`. Nothing is appended to the transcript in that case, so the turn
used to end silently (#629). The producer now stores the problem as
assistant text so the model sees the failure; `respond` recognises that
message to give a `--no-loop` session one retry turn.

`_respond_and_store` appends consecutive assistant `message` chunks to one
transcript entry, so narration streamed before the report is prepended to it.
The stored text therefore need not *start* with the prefix; the prefix is
distinctive enough to look for anywhere in the message.
"""

TOOL_CALL_ERROR_PREFIX = "**Error:** Tool call"


def is_tool_call_error(message):
    """Whether an LMC message contains the assistant text reporting an unrunnable tool call."""
    return (
        message.get("role") == "assistant"
        and message.get("type") == "message"
        and TOOL_CALL_ERROR_PREFIX in str(message.get("content", ""))
    )
