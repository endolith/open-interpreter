def truncate_output(data, max_output_chars=2800, spill_note=None):
    """Trim console output to the middle, keeping the head and tail.

    ``spill_note``, when given, is a sentence from the caller describing where
    the full untruncated text was saved. It is included in the message so the
    model can go and read the part that was cut instead of having to re-run the
    command.
    """
    needs_truncation = False

    # Calculate how much to show from start and end
    chars_per_end = max_output_chars // 2

    message = (f"Output truncated ({len(data):,} characters total). "
               f"Showing {chars_per_end:,} characters from start/end. ")
    if spill_note:
        message += spill_note + " "
    message += ("To handle large outputs, store result in python var first "
                "`result = command()` then process that, search with `result.find('text')`, "
                "repeat shell commands with wc/grep/sed, or read the file in small chunks, break it down "
                "into smaller steps, etc.\n\n")

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
