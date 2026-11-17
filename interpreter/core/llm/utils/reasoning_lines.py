"""Remove hollow "go" lines from model reasoning.

Some models pad their thinking with empty directives -- a line that only says
"Run.", "Execute.", "Let's do it." and nothing else. They add no information to
the transcript, and because stored reasoning is echoed back on later turns, the
model learns to keep producing them.

Only whole lines that consist of such a directive are removed. A line that adds
real content ("Run the tests after editing") is left untouched, as is every
other line.
"""

import re

# Normalized (lower-case, no trailing punctuation) phrases that are only an
# instruction to proceed and carry no reasoning content.
_FILLER_PHRASES = frozenset(
    {
        "run",
        "run it",
        "run that",
        "run this",
        "execute",
        "execute it",
        "execute that",
        "execute this",
        "do it",
        "just do it",
        "lets do it",
        "let us do it",
        "lets run",
        "lets run it",
        "let us run it",
        "lets execute",
        "lets execute it",
        "let us execute it",
        "go",
        "go ahead",
        "go for it",
        "proceed",
        "lets go",
        "let us go",
        "lets start",
        "lets begin",
        "here goes",
        "now run it",
        "now execute",
        "now execute it",
        "ok",
        "okay",
        "alright",
        "all right",
        "sure",
        "sounds good",
        "will do",
        "ok lets do it",
        "okay lets do it",
        "alright lets do it",
    }
)

# Leading blockquote markers, bullets, and numbered-list prefixes. A filler
# line is still filler when the model marks it as a list item.
_LEADING_MARKERS = re.compile(r"^\s*(?:>\s*)*(?:[-*\u2022]\s*|\d+[.)]\s*)?")


def _normalize(text):
    """Return the comparison form of ``text``: case-folded, no markers or punctuation."""
    text = _LEADING_MARKERS.sub("", text.strip())
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("'", "")
    text = re.sub(r"\s+", " ", text).lower()
    text = text.strip("*_~`\"' ")
    return text.rstrip(" .!,;:")


def is_filler_line(line):
    """True when ``line`` is nothing but a directive to run/execute/proceed."""
    return _normalize(line) in _FILLER_PHRASES


def _can_still_become_filler(text):
    """True while the partial line ``text`` is still a prefix of a filler phrase.

    Used by the streaming filter: text that can no longer become filler is
    emitted immediately, so ordinary reasoning is not delayed.
    """
    raw = text.strip()
    # A bare digit run is still a numbered-list marker, not content.
    if re.fullmatch(r"(?:>\s*)*\d+[.)]?", raw):
        return True

    candidate = _normalize(text)
    return any(phrase.startswith(candidate) for phrase in _FILLER_PHRASES)


def strip_filler_reasoning_lines(text):
    """Drop filler-only lines from ``text``, preserving every other byte."""
    if not isinstance(text, str) or not text:
        return text
    line_filter = FillerLineFilter()
    cleaned = line_filter.feed(text) + line_filter.flush()
    return text if cleaned == text else cleaned


class FillerLineFilter:
    """Streaming version of :func:`strip_filler_reasoning_lines`.

    Feed reasoning deltas as they arrive and emit the returned text. Complete
    non-filler lines are emitted immediately; a line is held only while it could
    still turn out to be a filler directive. Call :meth:`flush` when the stream
    ends to emit any held remainder.
    """

    def __init__(self):
        self._line = ""
        self._passthrough = False

    def feed(self, text):
        """Return the portion of ``text`` that is safe to show, holding at most one partial line."""
        if not text:
            return ""

        out = []
        for char in text:
            if self._passthrough:
                out.append(char)
                if char == "\n":
                    self._passthrough = False
                continue

            self._line += char

            if char == "\n":
                line, self._line = self._line, ""
                if not is_filler_line(line):
                    out.append(line)
                continue

            if not _can_still_become_filler(self._line):
                # This line cannot become a filler directive, so stream it now.
                out.append(self._line)
                self._line = ""
                self._passthrough = True

        return "".join(out)

    def flush(self):
        """Return the held remainder, unless it is itself a filler directive."""
        line, self._line = self._line, ""
        self._passthrough = False
        if line and not is_filler_line(line):
            return line
        return ""
