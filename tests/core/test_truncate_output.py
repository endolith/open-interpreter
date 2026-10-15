"""truncate_output keeps head/tail but summarizes the hole in the middle.

Console output over max_output is stored as head + marker + tail. The marker
used to be a bare "[...]", which told the model nothing about what was cut.
It now carries a structural summary -- size plus either the dominant repeated
line or an orienting sample -- so the model can judge whether the middle is
worth fetching instead of guessing.
"""

from interpreter.core.utils.truncate_output import truncate_output


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
