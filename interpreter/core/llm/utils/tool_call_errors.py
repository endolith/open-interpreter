"""Marker for the assistant message that reports a tool call we could not run.

`run_tool_calling_llm` can receive a tool call it cannot turn into code —
an unknown function name, or `execute` without both `language` and
`code`. Nothing is appended to the transcript in that case, so the turn
used to end silently (#629). The producer now stores the problem as
assistant text so the model sees the failure; `respond` recognises that
message to give a `--no-loop` session one retry turn.
"""

TOOL_CALL_ERROR_PREFIX = "**Error:** Tool call"


def is_tool_call_error(message):
    """Whether an LMC message is the assistant text reporting an unrunnable tool call."""
    return (
        message.get("role") == "assistant"
        and message.get("type") == "message"
        and str(message.get("content", "")).startswith(TOOL_CALL_ERROR_PREFIX)
    )
