"""truncate_output keeps head/tail but summarizes the hole in the middle.

Console output over max_output is stored as head + marker + tail. The marker
used to be a bare "[...]", which told the model nothing about what was cut.
It now carries a structural summary -- size plus either the dominant repeated
line or an orienting sample -- so the model can judge whether the middle is
worth fetching instead of guessing.
"""

from interpreter.core.utils.truncate_output import truncate_output

import re


def _stream(chunks, max_output_chars=2800):
    """Feed chunks the way the streaming loop does: append, then truncate."""
    stored = ""
    for chunk in chunks:
        stored += chunk
        stored = truncate_output(stored, max_output_chars=max_output_chars)
    return stored


def _marker_of(stored):
    match = re.search(r"\[\.\.\. .*?\.\.\.\]", stored)
    assert match is not None, "expected a truncation marker in the output"
    return match.group(0)


def _marker_lines(marker):
    match = re.search(r"\((\d+) lines?\)", marker)
    return int(match.group(1))


def test_short_output_passes_through_untouched():
    """Outputs under the cap must come back byte-identical, header included."""
    assert truncate_output("small output", max_output_chars=100) == "small output"


def test_head_and_tail_are_preserved():
    """The start (what ran) and end (result/traceback) must survive."""
    data = "HEAD-" + "x" * 500 + "-TAIL"
    out = truncate_output(data, max_output_chars=100)
    assert out.startswith("Output truncated")
    assert "HEAD-" in out and "-TAIL" in out


def test_repetitive_middle_names_the_repeated_line():
    """Progress-spam middles must be identified, not just counted."""
    middle = "Epoch 7/50 - loss: 0.317\n" * 40
    data = "START\n" + middle + "DONE\n"
    out = truncate_output(data, max_output_chars=100)
    assert "mostly repeated" in out
    assert "Epoch 7/50 - loss: 0.317" in out
    # Edge fragments where the cut lands mid-line don't disturb the count.
    assert "lines) omitted" in out


def test_varied_middle_shows_an_orienting_sample():
    """A non-repetitive middle must show where it starts, bounded in size."""
    middle = "".join(f"step {i}: a completely different log line here\n" for i in range(40))
    data = "START\n" + middle + "DONE\n"
    out = truncate_output(data, max_output_chars=100)
    assert "starts with:" in out
    # The cut lands on a line boundary, so the sample skips the blank stub
    # and shows the first content line of the omitted region.
    assert "step 1: a completely different log line here" in out
    assert "characters (" in out and "lines) omitted" in out


def test_single_giant_line_middle_is_capped():
    """A minified-style one-line middle must not leak unbounded text."""
    data = "A" * 60 + "B" * 5000 + "C" * 60
    out = truncate_output(data, max_output_chars=100)
    assert "1 line" in out
    assert len(out) < 2800 + 500


def test_retruncation_does_not_stack_headers():
    """Running the stored (already truncated) output through again stays single."""
    data = "START\n" + "".join(f"log line {i}\n" for i in range(100)) + "DONE\n"
    once = truncate_output(data, max_output_chars=100)
    twice = truncate_output(once, max_output_chars=100)
    assert twice.count("Output truncated") == 1


def test_streaming_totals_accumulate_across_passes():
    """Re-truncation must update the same summary, not restart it.

    Ten 1005-char chunks arrive one at a time with truncation after each,
    the way the streaming loop stores console output. The final marker must
    account for everything cut since the first pass (~7k chars), not just
    the last pass, and quote output -- never an older marker.
    """
    chunks = [f"log line {i:05d} " + "x" * 990 + "\n" for i in range(10)]
    fed = "".join(chunks)
    stored = _stream(chunks)
    assert stored.count("Output truncated") == 1
    marker = _marker_of(stored)
    assert marker.count("[...") == 1
    true_omitted = fed[1400:-1400]
    assert f"{len(true_omitted):,} characters" in marker
    # Chars are exact; lines overcount by ~1 per pass here because 1005-char
    # lines never align with the 1400-char windows, so every re-slice
    # re-counts edge fragments. Realistic short-line output stays exact.
    assert abs(_marker_lines(marker) - len(true_omitted.splitlines())) <= len(chunks)
    assert stored[stored.index("Output truncated") :].startswith(
        "Output truncated"
    )
    assert "log line 00000" in stored  # head is the true start, stably kept
    assert "log line 00009" in stored  # tail tracks the newest chunk


def test_streaming_preserves_the_repetition_verdict():
    """A progress-spam stream must still read as progress spam at the end.

    The qualitative clause is carried forward across passes (the middle only
    grows at its end), so the final marker names the repeated line instead
    of re-deriving -- and misquoting -- from degraded text.
    """
    chunk = "Epoch 7/50 - loss: 0.317\n" * 40
    fed = chunk * 5
    stored = _stream([chunk] * 5)
    marker = _marker_of(stored)
    assert "mostly repeated" in marker
    assert "Epoch 7/50 - loss: 0.317" in marker
    true_omitted = fed[1400:-1400]
    assert f"{len(true_omitted):,} characters" in marker
    assert abs(_marker_lines(marker) - len(true_omitted.splitlines())) <= 4


def test_nested_markers_fall_back_to_a_single_fresh_summary():
    """Corrupt input (a marker quoting a marker) must not nest further.

    Totals from nested markers can't be trusted, so every marker span is
    dropped and the remainder is summarized fresh -- coherent and single,
    even if the count no longer reaches the original output.
    """
    inner = "[... 5 characters (1 line) omitted ...]"
    data = "A" * 1400 + f"\n{inner}\n{inner}\n" + "B" * 1400
    out = truncate_output(data, max_output_chars=100)
    assert out.count("Output truncated") == 1
    assert out.count("[...") == 1
