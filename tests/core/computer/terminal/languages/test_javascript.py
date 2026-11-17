import io

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


def test_line_postprocessor_stops_stripping_pipes_once_execution_starts():
    """Program output that starts with four "| " tokens survives execution start.

    Node 24's continuation prompts only precede the echoed buffered source; the
    wrapper's first active-line marker is printed before the program runs. A
    printed markdown row such as "| | | | value |" arrives after that marker, so
    treating its cells as prompts would silently delete them. A later run
    echoes its input again, which restarts the prompt phase.
    """
    js = JavaScript()
    assert js.line_postprocessor("| | | | ##active_line1##\n") == (
        "##active_line1##\n"
    )
    assert js.line_postprocessor("| | | | value |\n") == "| | | | value |\n"
    js.preprocess_code("console.log('next')")
    assert js.line_postprocessor("| | | | echoed\n") == "echoed\n"


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


def test_stderr_pipe_prefix_survives_the_echo_phase():
    """A markdown row written to stderr keeps its pipes while stdout echoes source.

    Node's continuation prompts are part of the stdout echo, so the prompt
    cleanup must not touch stderr. stderr can be read first while the echo is
    still pending; stripping it then would corrupt the program's own output.
    """
    js = JavaScript()
    js.handle_stream_output(io.StringIO("| | | | value |\n"), True)
    assert js.output_queue.get_nowait()["content"] == "| | | | value |\n"


def test_stdout_pipe_prefix_is_still_stripped_during_the_echo_phase():
    """The continuation-prompt cleanup continues to apply to stdout echo lines."""
    js = JavaScript()
    js.handle_stream_output(io.StringIO("| | | | echoed\n"), False)
    assert js.output_queue.get_nowait()["content"] == "echoed\n"


def test_stderr_marker_like_output_does_not_end_the_guard_phase():
    """stderr text that looks like a marker must not end the stdout echo phase.

    Only the wrapper's own stdout markers delimit the echo, so a program
    printing "##active_line" to stderr must not make the next stdout echo line
    skip prompt cleanup.
    """
    js = JavaScript()
    js.handle_stream_output(io.StringIO("##active_line9##\n"), True)
    js.handle_stream_output(io.StringIO("| | | | echoed\n"), False)
    contents = []
    while not js.output_queue.empty():
        contents.append(js.output_queue.get_nowait()["content"])
    # The stdout echo line was still cleaned up, so stderr did not end the phase.
    assert "echoed\n" in contents


def test_literal_prompt_prefix_after_execution_is_preserved():
    """Program output that begins with "> ..." keeps its literal prefix.

    ``console.log("> ... details")`` prints exactly that after the wrapper's
    active-line marker has ended the echo phase. Stripping prompt tokens there
    would silently delete the program's own text, leaving only "details\\n".
    The prompt cleanup is therefore confined to the echoed input.
    """
    js = JavaScript()
    js.handle_stream_output(
        io.StringIO("##active_line1##\n> ... details\n"), is_error_stream=False
    )
    js.output_queue.get_nowait()  # the active-line chunk
    assert js.output_queue.get_nowait()["content"] == "> ... details\n"


def test_prompt_only_program_line_is_preserved():
    """A program that prints just "> " keeps it once execution has started.

    Prompt tokens only belong to the echoed source. A bare prompt-only line
    that arrives after the active-line marker is the program's own output, so
    dropping the whole line would silently delete it.
    """
    js = JavaScript()
    js.handle_stream_output(io.StringIO("##active_line1##\n"), is_error_stream=False)
    js.handle_stream_output(io.StringIO("> \n"), is_error_stream=False)
    contents = []
    while not js.output_queue.empty():
        contents.append(js.output_queue.get_nowait()["content"])
    assert contents == [1, "> \n"]


def test_stderr_prompt_only_line_is_preserved():
    """stderr is never the REPL's echo, so a bare "> " there is program output."""
    js = JavaScript()
    js.handle_stream_output(io.StringIO("> \n"), is_error_stream=True)
    assert js.output_queue.get_nowait()["content"] == "> \n"
    # Even the newline-free form is program text on stderr, not node's prompt.
    js2 = JavaScript()
    js2.handle_stream_output(io.StringIO(">"), is_error_stream=True)
    assert js2.output_queue.get_nowait()["content"] == ">"


def test_live_prompt_without_a_newline_is_dropped():
    """Node's final prompt arrives without a separator and is dropped whole.

    Readline writes the prompt before reading input, so when a run ends the
    prompt reaches the reader as a ">"-only line without a newline. Program
    output that merely resembles a prompt comes from console.log, ends with a
    newline, and is kept.
    """
    js = JavaScript()
    js.handle_stream_output(io.StringIO("##active_line1##\n"), is_error_stream=False)
    js.handle_stream_output(io.StringIO(">"), is_error_stream=False)
    contents = []
    while not js.output_queue.empty():
        contents.append(js.output_queue.get_nowait()["content"])
    assert contents == [1]


def test_multiline_code_marks_the_start_of_execution():
    """Multi-line code gets one start marker where per-line markers are unsafe.

    A console.log between the lines of a bracketed block would split brackets
    and object literals, so the wrapper emits a single marker before the block
    instead. Without it the echo phase never ends when the REPL does not print
    active-line markers, and a program line starting with "> " loses its prefix.
    """
    code = preprocess_javascript("function f() {\n  console.log('x');\n}\nf()")
    assert "##start_of_execution##" in code
    assert "##active_line" not in code
    assert "##end_of_execution##" in code


def test_start_marker_ends_the_echo_phase_without_being_displayed():
    """The multi-line start marker ends prompt cleanup and is not shown as output."""
    js = JavaScript()
    js.handle_stream_output(
        io.StringIO("##start_of_execution##\n> ... details\n"), is_error_stream=False
    )
    contents = []
    while not js.output_queue.empty():
        contents.append(js.output_queue.get_nowait()["content"])
    assert contents == ["> ... details\n"]


def test_stderr_prompt_prefix_is_preserved_during_the_echo_phase():
    """stderr that starts with a prompt-like "> " is program output, not echo.

    Only stdout carries the REPL's echoed input, so prompt cleanup must leave
    stderr untouched even while the echo phase is active.
    """
    js = JavaScript()
    js.handle_stream_output(io.StringIO("> hello\n"), is_error_stream=True)
    assert js.output_queue.get_nowait()["content"] == "> hello\n"


def test_marker_echo_keeps_no_prompt_after_a_multiline_block():
    """The echoed end marker must not leave a stray prompt token behind.

    A multi-line block ends the echo phase through its start marker, so the
    echoed end-of-execution line is no longer cleaned by the echo gate. The
    reader strips the marker from that line, and a prompt token left in front
    of it would surface as a bare ">" chunk, so the tokens are removed before
    the gate has a chance to skip them.
    """
    js = JavaScript()
    js._echoing_input = True
    assert js.line_postprocessor("##start_of_execution##\n") is None
    assert js._echoing_input is False
    assert js.line_postprocessor("> ##end_of_execution##\n") == "##end_of_execution##\n"
