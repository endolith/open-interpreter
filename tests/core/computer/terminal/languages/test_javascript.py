from interpreter.core.computer.terminal.languages.javascript import (
    JavaScript,
    preprocess_javascript,
)


def test_preprocess_javascript_adds_end_marker():
    """preprocess_javascript() wraps code in try/catch and adds an end-of-execution marker."""
    code = preprocess_javascript("console.log(1)")
    assert "##end_of_execution##" in code
    assert "try {" in code


def test_preprocess_single_line_adds_active_line_markers():
    """preprocess_javascript() inserts ##active_lineN## markers before each source line."""
    code = preprocess_javascript("a()\nb()")
    assert "##active_line1##" in code
    assert "##active_line2##" in code


def test_line_postprocessor_filters_node_banner():
    """JavaScript line_postprocessor drops Node.js welcome banners but keeps real output."""
    js = JavaScript()
    assert js.line_postprocessor("Welcome to Node.js v20") is None
    assert js.line_postprocessor("actual output") == "actual output"


def test_start_cmd_is_node_interactive():
    """JavaScript starts the node REPL via ["node", "-i"].

    A wrong binary or flag would fail to start the language subprocess.
    """
    assert JavaScript().start_cmd == ["node", "-i"]


def test_line_postprocessor_drops_undefined_and_help():
    """The REPL's "undefined" echo and help banner are dropped.

    Both literals are compared after strip(); a renamed or case-changed literal
    would leak REPL noise into the model's view.
    """
    js = JavaScript()
    assert js.line_postprocessor("undefined") is None
    assert js.line_postprocessor('Type ".help" for more information.') is None


def test_line_postprocessor_strips_leading_prompt_arrows():
    """Leading "> " REPL prompts are removed from output lines.

    The regex strips one or more leading ">" tokens; without it every echoed
    value would keep the prompt marker.
    """
    js = JavaScript()
    assert js.line_postprocessor("> > 42") == "42"


def test_line_postprocessor_strips_continuation_prompts():
    """Node's multi-line continuation prompts are removed from result lines.

    Node echoes a buffered multi-line command with repeated "... " prompts.
    Stripping only the ">" runs left those dots glued to the first real output
    line (e.g. "... ... ... ##active_line1##"), so the model saw prompt noise
    as part of the program's output.
    """
    js = JavaScript()
    assert (
        js.line_postprocessor("> > ... ... ... ... ... ##active_line1##\n")
        == "##active_line1##"
    )


def test_line_postprocessor_strips_node24_multiline_prompt():
    """Node 24's "| " continuation prompts are stripped without eating table rows.

    Node 24's readline uses "| " for a buffered multi-line command's
    continuation prompt, so a multi-line wrapper leaked a run of "| " onto the
    result line. A single or double leading "| " is legitimate output (a
    markdown table row), so only runs of four or more are stripped.
    """
    js = JavaScript()
    assert (
        js.line_postprocessor("> > | | | | hello from node v24\n")
        == "hello from node v24"
    )
    assert js.line_postprocessor("| a | b |\n") == "| a | b |"
    assert js.line_postprocessor("| | empty\n") == "| | empty"
    assert js.line_postprocessor("| | | three\n") == "| | | three"


def test_line_postprocessor_keeps_table_row_delimiter_after_prompts():
    """A table row printed after Node 24's prompt run keeps its leading delimiter.

    Node 24's continuation prompt and a row's first cell delimiter are both
    "| ", so stripping any long prompt run also consumes the row's own
    delimiter and turns "| a | b |" into "a | b |", which no longer renders as
    a table. These calls exercise the run-length fallback used when the prompt
    count is unknown: line_postprocessor puts one delimiter back when the
    remainder still ends with a cell delimiter and contains another one.
    """
    js = JavaScript()
    assert js.line_postprocessor("> > " + "| " * 7 + "| a | b |\n") == "| a | b |"
    assert js.line_postprocessor("> > " + "| " * 9 + "| 1 | 2 | 3 |\n") == "| 1 | 2 | 3 |"


def test_line_postprocessor_preserves_leading_empty_cells_after_prompts():
    """A row with leading empty cells survives the counted prompt strip.

    preprocess_code counts the prompts Node will print, so line_postprocessor
    removes exactly that many "| " tokens and leaves the row's own delimiters
    untouched. Without the count, the run-length fallback cannot tell the
    prompts from the row and drops its empty leading cells.
    """
    js = JavaScript()
    processed = js.preprocess_code('if (true) { console.log("| | | a | b |") }')
    prompts = js._continuation_prompts
    assert prompts >= 4
    line = "> > " + "| " * prompts + "| | | a | b |"
    assert js.line_postprocessor(line) == "| | | a | b |"


def test_line_postprocessor_keeps_leading_empty_cells_after_node22_dot_prompts():
    """Node 22 dot prompts must not trigger Node 24's "| " prompt stripping.

    Node 22 renders a buffered multi-line command as "> > ... ... ... ... "
    and glues the first result line to the same line. The "| " delimiters
    after those dots are the printed table row, not prompt tokens; stripping
    the counted run would eat the row's leading empty cells and turn
    "| | | | a | b |" into "a | b |".
    """
    js = JavaScript()
    js.preprocess_code('if (true) { console.log("| | | | a | b |") }')
    prompts = js._continuation_prompts
    assert prompts >= 1
    line = "> > " + "... " * prompts + "| | | | a | b |\n"
    assert js.line_postprocessor(line) == "| | | | a | b |"


def test_preprocess_code_counts_lines_like_node_readline():
    """Form feeds are not line breaks for Node's prompt count.

    Python's splitlines() splits on form feed, so a form feed inside the code
    inflated the prompt count and the counted strip removed one delimiter too
    many from a Node 24 prompt line. Node's readline only splits on CR, LF and
    CRLF, so the count must use those rules.
    """
    js = JavaScript()
    js.preprocess_code('if (true) {\f console.log("| | | a | b |") }')
    assert js._continuation_prompts == 4
    line = "> > " + "| " * 4 + "| | | a | b |"
    assert js.line_postprocessor(line) == "| | | a | b |"


def test_line_postprocessor_keeps_result_dots_after_node22_dot_prompts():
    """A result beginning with dots survives the Node 22 prompt strip.

    Node 22 glues the printed result right after the last "... " prompt,
    so stripping every dot run also ate the result's own leading dots
    ("...waiting" became "waiting"). Strip only the recorded prompt count.
    """
    js = JavaScript()
    js.preprocess_code('if (true) { console.log("...waiting") }')
    prompts = js._continuation_prompts
    assert prompts >= 1
    line = "> > " + "... " * prompts + "...waiting\n"
    assert js.line_postprocessor(line) == "...waiting"


def test_line_postprocessor_falls_back_when_dot_prompt_count_is_stale():
    """A recorded dot-prompt count that no longer matches falls back to a wholesale strip.

    preprocess_code records the prompt count for the code it just wrapped. If a
    later call carries a stale count (Node changed how it renders the run, or a
    caller invoked line_postprocessor outside a completed run) the counted
    strip removes nothing, and leaving the dots would leak the prompt run into
    the model's output. Falling back to the wholesale dot strip keeps the
    prompt noise out even though the count is unusable.
    """
    js = JavaScript()
    js._continuation_prompts = 5
    assert js.line_postprocessor("> > ... ... ...waiting\n") == "waiting"


def test_line_postprocessor_falls_back_when_pipe_prompt_count_is_stale():
    """A mismatched "| " prompt count falls through to the run-length heuristic.

    Node 24 prints one "| " per buffered line, so the counted strip is exact.
    When the recorded count no longer matches what Node printed, the counted
    strip matches nothing and the code must fall through to the run-length
    fallback instead of returning the line with its prompt run still attached.
    """
    js = JavaScript()
    js._continuation_prompts = 6
    assert js.line_postprocessor("> > | | | | | hello\n") == "hello"


def test_line_postprocessor_leaves_unprompted_table_rows_alone():
    """A later output row's leading empty cells are not prompt tokens.

    Continuation prompts appear only on the line Node glued its prompt run to,
    the one carrying a ">" prefix. Treating any long "| " run as prompts ate
    the empty leading cells of a later row, so lines without the prefix are
    passed through unchanged.
    """
    js = JavaScript()
    assert js.line_postprocessor("| | | | a | b |\n") == "| | | | a | b |"
    assert (
        js.line_postprocessor("| | | | | 1 | 2 | 3 |\n")
        == "| | | | | 1 | 2 | 3 |"
    )


def test_line_postprocessor_strips_dot_space_and_newline():
    """Leading/trailing dots, spaces and newlines are trimmed.

    The strip charset is ". \\n"; dropping it would leave padding in the output.
    """
    js = JavaScript()
    assert js.line_postprocessor("  .value. ") == "value"


def test_preprocess_javascript_multiline_skips_active_line_markers():
    """Code containing brackets is treated as multiline and gets no line markers.

    The guard looks for any of { } [ ]; a mutated set would either inject
    markers into multiline code or skip them for single lines.
    """
    # Only an opening brace: if the "{" pattern were renamed, this would be
    # treated as single-line and receive markers.
    code = preprocess_javascript("function f() {\n  return 1;\n")
    assert "##active_line" not in code
    assert "##end_of_execution##" in code
