def truncate_output(data, max_output_chars=2800, spill_note=None):
    """Trim console output to the middle, keeping the head and tail.

    ``spill_note``, when given, is a sentence from the caller describing where
    the full untruncated text was archived. It comes last, after the advice to
    ask a narrower question in the first place: re-reading a megabyte of output
    is almost never what the model actually needs, and the note is there for the
    cases where it genuinely is.
    """
    needs_truncation = False

    # Calculate how much to show from start and end
    chars_per_end = max_output_chars // 2

    message = (f"Output truncated ({len(data):,} characters total). "
               f"Showing {chars_per_end:,} characters from start/end. "
               "Prefer re-running the command shaped to the answer you need "
               "rather than reading all of it: `cmd > /dev/null 2>&1 && echo OK "
               "|| echo FAILED` when you only need pass/fail, or pipe through "
               "grep/wc -l/head/tail/jq for a specific value. In Python, keep "
               "the result in a variable and inspect that "
               "(`result = command()`, then `result.find('text')`). ")
    if spill_note:
        message += spill_note + " "
    message += "\n\n"

    # Remove previous truncation message if it exists
    if data.startswith(message):
        data = data[len(message) :]
        needs_truncation = True

    # If data exceeds max length, truncate it and add message
    if len(data) > max_output_chars or needs_truncation:
        first_part = data[:chars_per_end]
        last_part = data[-chars_per_end:]
        data = message + first_part + "\n[...]\n" + last_part

    return data
