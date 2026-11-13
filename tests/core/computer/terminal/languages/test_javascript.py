from interpreter.core.computer.terminal.languages.javascript import (
    JavaScript,
    _JS_FRAME_MARKER,
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


def test_line_postprocessor_strips_dot_space_and_newline():
    """Leading/trailing dots, spaces and newlines are trimmed.

    The strip charset is ". \\n"; dropping it would leave padding in the output.
    """
    js = JavaScript()
    assert js.line_postprocessor("  .value. ") == "value"


def test_program_output_after_the_frame_is_returned_unchanged():
    """Output printed by the program must survive prompt cleanup untouched.

    Before the frame, prompt glyphs, dots and padding are Node's scaffolding.
    After the frame, the same leading characters can legitimately be program
    output (" > answer", "...", an indented or padded line), so the
    postprocessor must return the line verbatim.
    """
    js = JavaScript()
    assert js.line_postprocessor(_JS_FRAME_MARKER) is None
    for line in [
        "> 42",
        "...continued",
        "  indented output",
        ".leading dot",
        "| | table row",
    ]:
        assert js.line_postprocessor(line) == line


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


def test_line_postprocessor_strips_node24_multiline_prompt():
    """Node 24 echoes buffered multi-line input with a run of "| " prompts.

    Node 22 used "... "; Node 24's readline uses kMultilinePrompt "| ", so the
    prompt leaked onto the result line ("| | | | hello"). The run is one prompt
    per buffered input line, so before the frame any leading bar run is
    scaffolding; after it, bars are program output (a markdown table row,
    `| | empty`) and stay.
    """
    js = JavaScript()
    assert js.line_postprocessor("| | | | hello from node v24") == "hello from node v24"
    assert js.line_postprocessor("| | | three") == "three"

    js = JavaScript()
    js.preprocess_code("console.log(1)")
    assert js.line_postprocessor("__oi_js_frame__") is None
    assert js.line_postprocessor("| a | b |") == "| a | b |"
    assert js.line_postprocessor("| | empty") == "| | empty"
    assert js.line_postprocessor("| | | three") == "| | | three"


def test_line_postprocessor_keeps_output_bars_after_the_frame_marker():
    """Once the frame marker has printed, leading bars are program output.

    Node 24 puts its continuation prompts in front of the output, so without a
    frame the rule that strips four prompts also strips bars the program
    printed: `if (true) { console.log("| | empty"); }` came back as `empty`.
    The marker separates the echo of our own input from everything the program
    prints, so only the echo is prompt-stripped.
    """
    js = JavaScript()

    processed = js.preprocess_code("console.log(1)")
    assert "__oi_js_frame__" in processed

    # Before the marker: Node echoing our buffered input.
    assert js.line_postprocessor("| | | | hello from node v24") == "hello from node v24"
    # The marker itself is scaffolding, not output.
    assert js.line_postprocessor("__oi_js_frame__") is None
    # After the marker: whatever the program printed, bars included.
    assert js.line_postprocessor("| | empty") == "| | empty"
    assert js.line_postprocessor("| a | b |") == "| a | b |"


def test_a_prefixed_frame_marker_is_recognized_and_output_bars_survive():
    """Node may print the marker behind `>` and `| ` prompts.

    The marker was compared before any prompt stripping, so a prefixed marker
    went unrecognised: it leaked into the output, the frame never flipped, and
    later output beginning with four bars was stripped.
    """
    js = JavaScript()
    processed = js.preprocess_code("console.log(1)")
    assert "__oi_js_frame__" in processed

    assert js.line_postprocessor("> | | | | __oi_js_frame__") is None
    assert js.line_postprocessor("| | | | output bars") == "| | | | output bars"


def test_line_postprocessor_recognizes_node22_continuation_prompts():
    """Node <= 23 uses "... " as readline's continuation prompt; Node 24 uses "| ".

    The marker has to be recognized behind either glyph. Keyed to "| " only, a
    Node 22 run leaves the prefixed marker in the output and never starts the
    frame: verified against v22.22.1, which emits
    "> > ... ... ... __oi_js_frame__" ahead of the first result.
    """
    js = JavaScript()
    assert "__oi_js_frame__" in js.preprocess_code("console.log(1)")

    assert js.line_postprocessor("> > ... ... ... __oi_js_frame__") is None
    # The frame is open, so program output that starts with bars survives.
    assert js.line_postprocessor("| | empty") == "| | empty"
