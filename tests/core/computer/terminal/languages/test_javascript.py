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
    """Node's multi-line continuation prompts are removed, not leaked as output.

    The JS wrapper is fed to `node -i` as a multi-line try/catch, so the REPL
    echoes "> " and "... " prompts before each buffered line. Only the prompt
    tokens may be stripped; the echoed code and marker must survive intact.
    """
    js = JavaScript()
    assert (
        js.line_postprocessor("> > ... ... ... ... ... ##active_line1##\n")
        == "##active_line1##\n"
    )


def test_line_postprocessor_strips_node24_multiline_prompt():
    """Node 24 echoes buffered multi-line input with a run of "| " prompts.

    Node 22 used "... "; Node 24's readline uses kMultilinePrompt "| ", so the
    prompt leaked onto the result line ("| | | | hello"). A single or double
    leading "| " is legitimate output (a markdown table row), so only runs of
    four or more are stripped.
    """
    js = JavaScript()
    assert js.line_postprocessor("| | | | hello from node v24\n") == (
        "hello from node v24\n"
    )
    assert js.line_postprocessor("| a | b |\n") == "| a | b |\n"
    assert js.line_postprocessor("| | empty\n") == "| | empty\n"
    assert js.line_postprocessor("| | | three\n") == "| | | three\n"


def test_line_postprocessor_preserves_trailing_newline():
    """Trailing newlines are kept so non-streaming runs do not join adjacent lines.

    Terminal.run(stream=False) concatenates output chunks, so stripping the
    newline would glue the next line onto the previous one ("42" onto "js ok |").
    """
    js = JavaScript()
    assert js.line_postprocessor("js ok | v22.22.1\n") == "js ok | v22.22.1\n"


def test_line_postprocessor_preserves_newline_on_prompt_only_lines():
    r"""A prompt-only echoed line keeps its newline instead of collapsing to "".

    Prompt tokens may be followed by spaces or tabs but not by the line
    separator; a `\s` class would consume the newline, so an empty echoed line
    would lose its break and the next chunk would be joined onto it.
    """
    js = JavaScript()
    assert js.line_postprocessor("> \n") == "\n"
    assert js.line_postprocessor(">>> \n") == "\n"
    assert js.line_postprocessor("... \n") == "\n"


def test_line_postprocessor_preserves_non_prompt_text():
    """Text that does not start with a REPL prompt is left verbatim.

    The old implementation stripped ". \\n" from both ends of every line, which
    silently deleted legitimate trailing dots and padding. Only prompt tokens
    are removed now.
    """
    js = JavaScript()
    assert js.line_postprocessor("  .value. ") == "  .value. "
    assert js.line_postprocessor("table | value\n") == "table | value\n"


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


def test_active_line_marker_does_not_add_blank_output_line():
    """A marker printed on its own line must not add a leading blank output chunk.

    The wrapper emits `##active_lineN##\n` before each source line. Stripping
    the marker leaves "\n", and Terminal.run(stream=False) concatenates
    consecutive output chunks, so keeping it would turn `ok\n` into
    `\nok\n`. Text that shares the marker's line is still kept.
    """
    import io

    js = JavaScript()
    js.handle_stream_output(
        io.StringIO("##active_line1##\nok\n##active_line2## extra\n"),
        is_error_stream=False,
    )
    chunks = []
    while not js.output_queue.empty():
        chunks.append(js.output_queue.get())

    assert chunks == [
        {"type": "console", "format": "active_line", "content": 1},
        {"type": "console", "format": "output", "content": "ok\n"},
        {"type": "console", "format": "active_line", "content": 2},
        {"type": "console", "format": "output", "content": " extra\n"},
    ]
