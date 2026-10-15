"""Cap console output stored in history, summarizing what was cut.

The head/tail split is kept (the start usually names what ran, the end holds
the result or traceback), but the old bare "[...]" marker told the model
nothing about the hole in the middle. The marker now carries a structural
summary of the omitted text -- its size, and either its dominant repeated
line (progress bars, per-epoch logs) or its opening line as an orienting
sample -- so the model can judge whether the middle is worth fetching with
get_last_output() instead of guessing.
"""

from collections import Counter
import re

_SAMPLE_LINE_CAP = 160

# Matches a truncation header stamped by an earlier pass. The character total
# changes every pass (truncating shortens the text), so an exact startswith
# check never matches twice and headers would stack; match the template with
# the counts left variable instead.
_TRUNCATION_HEADER_RE = re.compile(
    r"^Output truncated \([\d,]+ characters total\)\. "
    r"Showing [\d,]+ characters from start/end\. "
    r"To handle large outputs, .*?smaller steps, etc\."
    r"( Run `get_last_output\(\)\[\d+:\d+\]` to see the first page\.)?\n\n",
    re.DOTALL,
)


def _summarize_omitted(middle):
    """One-line summary of the truncated-away middle of an output."""
    lines = middle.splitlines()
    noun = "line" if len(lines) == 1 else "lines"
    parts = [f"{len(middle):,} characters ({len(lines)} {noun}) omitted"]
    stripped = [line.strip() for line in lines if line.strip()]
    if stripped:
        top, count = Counter(stripped).most_common(1)[0]
        if count >= 3 and count / len(stripped) >= 0.5:
            if len(top) > _SAMPLE_LINE_CAP:
                top = top[:_SAMPLE_LINE_CAP] + "…"
            parts.append(f'mostly repeated: "{top}" (x{count} of {len(stripped)})')
        elif lines:
            sample = next((ln.strip() for ln in lines if ln.strip()), "")[:_SAMPLE_LINE_CAP]
            if sample:
                parts.append(f'starts with: "{sample}"')
    return "[... " + "; ".join(parts) + " ...]"


def truncate_output(data, max_output_chars=2800, add_scrollbars=False):
    # if "@@@DO_NOT_TRUNCATE@@@" in data:
    #     return data

    needs_truncation = False

    # Calculate how much to show from start and end
    chars_per_end = max_output_chars // 2

    message = (f"Output truncated ({len(data):,} characters total). "
               f"Showing {chars_per_end:,} characters from start/end. "
               "To handle large outputs, store result in python var first "
               "`result = command()` then process that, search with `result.find('text')`, "
               "repeat shell commands with wc/grep/sed, or read the file in small chunks, break it down "
               "into smaller steps, etc.\n\n")

    # This won't work because truncated code is stored in interpreter.messages :/
    # If the full code was stored, we could do this:
    if add_scrollbars:
        message = (
            message.strip()
            + f" Run `get_last_output()[0:{max_output_chars}]` to see the first page.\n\n"
        )
    # Then we have code in `terminal.py` which makes that function work. It should be a toolbox tool though to just access messages IMO. Or like, self.messages.

    # Remove previous truncation header if this output was truncated before.
    # The total shrinks with every pass, so match the template, not the text.
    prior = _TRUNCATION_HEADER_RE.match(data)
    if prior:
        data = data[prior.end() :]
        needs_truncation = True

    # If data exceeds max length, truncate it and add message
    if len(data) > max_output_chars or needs_truncation:
        first_part = data[:chars_per_end]
        last_part = data[-chars_per_end:]
        middle = data[chars_per_end : len(data) - chars_per_end]
        marker = _summarize_omitted(middle) if middle else "[...]"
        data = message + first_part + "\n" + marker + "\n" + last_part

    return data
