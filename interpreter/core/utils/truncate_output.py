"""Cap console output stored in history, summarizing what was cut.

The head/tail split is kept (the start usually names what ran, the end holds
the result or traceback), but the old bare "[...]" marker told the model
nothing about the hole in the middle. The marker now carries a structural
summary of the omitted text -- its size, and either its dominant repeated
line (progress bars, per-epoch logs) or its opening line as an orienting
sample -- so the model can judge whether the middle is worth fetching with
get_last_output() instead of guessing.

Truncation runs on every streamed chunk, so a long output is re-truncated
many times. Re-truncation UPDATES the prior summary instead of recomputing
from the already-truncated body: the head is stable across passes (content
only grows at the end), the old totals are parsed out of the prior marker
and the newly cut text is added on, and the qualitative clause is carried
forward. Recomputing would both understate the totals and quote the old
marker as if it were output.
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


# A prior marker, without assuming anything about what surrounds it. The
# tempered content can't span a nested "[... ", so with nested markers (from
# the pre-accumulation version quoting its own marker) each match is one
# innermost marker.
_TRUNCATION_MARKER_RE = re.compile(r"\[\.\.\. ((?:(?!\[\.\.\. ).)*?)\.\.\.\]", re.DOTALL)
_PRIOR_TOTALS_RE = re.compile(r"^([\d,]+) characters \((\d+) lines?\) omitted(.*?)$", re.DOTALL)


def _format_marker(chars, lines, suffix):
    """Build the marker line from totals plus a qualitative clause."""
    noun = "line" if lines == 1 else "lines"
    return f"[... {chars:,} characters ({lines} {noun}) omitted{suffix} ...]"


def _describe_middle(middle):
    """Qualitative clause for an omitted middle: '' or '; ...'."""
    lines = middle.splitlines()
    stripped = [line.strip() for line in lines if line.strip()]
    if not stripped:
        return ""
    top, count = Counter(stripped).most_common(1)[0]
    if count >= 3 and count / len(stripped) >= 0.5:
        if len(top) > _SAMPLE_LINE_CAP:
            top = top[:_SAMPLE_LINE_CAP] + "…"
        return f'; mostly repeated: "{top}" (x{count} of {len(stripped)})'
    sample = next((line.strip() for line in lines if line.strip()), "")[:_SAMPLE_LINE_CAP]
    if sample:
        return f'; starts with: "{sample}"'
    return ""


def _summarize_omitted(middle):
    """One-line summary of the truncated-away middle of an output."""
    lines = middle.splitlines()
    return _format_marker(len(middle), len(lines), _describe_middle(middle))


def _split_prior_marker(text):
    """Split out our own prior marker(s).

    Returns (body, prior, split_at): body is the text with marker spans
    removed (plus adjacent newlines, so no phantom blank line remains and no
    fragment lines are joined or split versus the true text), prior is
    (chars, lines, suffix) when exactly one parseable marker existed, else
    None, and split_at is where the single marker was so a fitting body can
    put it back in place. Several matches (nested markers from the
    pre-accumulation version) or an unparseable match means the totals can't
    be trusted: every span is dropped and the remainder is summarized fresh.
    """
    matches = list(_TRUNCATION_MARKER_RE.finditer(text))
    parsed = []
    for match in matches:
        totals = _PRIOR_TOTALS_RE.match(match.group(1))
        if totals:
            parsed.append(
                (match, int(totals.group(1).replace(",", "")), int(totals.group(2)), totals.group(3))
            )
    body = text
    split_at = len(text)
    for match in reversed(matches):
        start, end = match.start(), match.end()
        if start > 0 and body[start - 1] == "\n":
            start -= 1
        if end < len(body) and body[end] == "\n":
            end += 1
        if len(matches) == 1:
            split_at = start
        body = body[:start] + body[end:]
    if len(matches) == 1 and len(parsed) == 1:
        return body, parsed[0][1:], split_at
    return body, None, split_at


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
        # A prior pass left its marker behind: pull it out, keep its totals,
        # and accumulate the newly cut text onto them. Content only grows at
        # the end, so the head below is the true start and the old middle
        # only grew -- its qualitative clause stays valid.
        body, prior, split_at = _split_prior_marker(data)
        if prior is not None and len(body) <= max_output_chars:
            # Fits now (the cap was raised mid-stream): the middle is still
            # gone, so put the old summary back where it was.
            old_chars, old_lines, suffix = prior
            old_marker = _format_marker(old_chars, old_lines, suffix)
            return message + body[:split_at] + "\n" + old_marker + "\n" + body[split_at:]
        first_part = body[:chars_per_end]
        last_part = body[-chars_per_end:]
        middle = body[chars_per_end : len(body) - chars_per_end]
        if prior is None:
            marker = _summarize_omitted(middle) if middle else "[...]"
        else:
            # Accumulate: the head is stable and content only grows at the
            # end, so the old totals plus the newly cut text is the true
            # omission. Chars are exact. Lines are exact when cuts land on
            # line boundaries; otherwise edge fragments re-sliced by a later
            # pass overcount by ~1 per pass -- immaterial next to the total.
            old_chars, old_lines, suffix = prior
            if not suffix:
                suffix = _describe_middle(middle)
            marker = _format_marker(
                old_chars + len(middle), old_lines + len(middle.splitlines()), suffix
            )
        data = message + first_part + "\n" + marker + "\n" + last_part

    return data
