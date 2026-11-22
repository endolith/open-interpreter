import re
import shutil
import threading

import pytest

from interpreter.core.computer.terminal.languages.ruby import Ruby


def test_ruby_preprocess_code_wraps_in_begin_rescue():
    """Ruby code is wrapped in begin/rescue with active-line and completion markers.

    The completion markers are unique to the block, so they are matched through
    the instance rather than spelled as the historical fixed strings.
    """
    ruby = Ruby()
    result = ruby.preprocess_code("puts 1")
    assert "begin" in result
    assert re.fullmatch(r"##end_of_execution_[0-9a-f]{8}##", ruby._end_marker)
    assert re.fullmatch(r"##execution_error_[0-9a-f]{8}##", ruby._error_marker)
    assert ruby._end_marker in result
    assert ruby._error_marker in result
    assert re.fullmatch(r"[0-9a-f]{8}", ruby._nonce)
    assert f"##active_line1_{ruby._nonce}##" in result


def test_ruby_line_postprocessor_before_preprocess_does_not_crash():
    """A line can reach the postprocessor before preprocess_code() runs.

    run() preprocesses first, but the reader thread starts at session start and
    irb prints its banner immediately; a restarted session can also deliver
    lines before the next block is preprocessed. Reading an attribute that
    __init__ had not set raised AttributeError inside the reader, which aborts
    the stream and kills the session. Construction must leave it defined.
    """
    ruby = Ruby()

    assert ruby.code_line_count == 0
    assert ruby._end_marker == "##end_of_execution##"
    assert ruby._error_marker == "##execution_error##"
    # No block is running, so the banner is ordinary text rather than an echo.
    assert ruby.line_postprocessor("Switch to inspect mode.\n") == (
        "Switch to inspect mode.\n"
    )
    assert ruby.line_postprocessor("nil\n") is None


def test_ruby_completion_markers_are_unique_per_block():
    """Each block gets its own marker strings so echoed input cannot forge them.

    irb echoes the submitted source before running it, comments included, so a
    fixed marker taken from that source would end the block early. A fresh
    nonce per block means no submitted line can match the completion marker.
    """
    ruby = Ruby()
    first = ruby.preprocess_code("1+1")
    first_end = ruby._end_marker
    first_error = ruby._error_marker
    second = ruby.preprocess_code("2+2")

    assert first_end != ruby._end_marker
    assert first_error != ruby._error_marker
    assert first_end in first
    assert first_end not in second


def test_ruby_echoed_comment_matching_the_old_marker_cannot_end_the_block():
    """A user comment spelling the historical marker is skipped like any echo.

    The comment is echoed back verbatim before it is compiled, so matching the
    fixed ##end_of_execution## string used to close the window and report
    completion while irb had not run the rest of the block yet. Only the
    block's own marker may end it.
    """
    ruby = Ruby()
    processed = ruby.preprocess_code('##end_of_execution##\nputs "after"')
    start = ruby.code_line_count
    assert start > 0

    for raw in processed.splitlines():
        assert ruby.line_postprocessor(raw + "\n") is None

    assert ruby.code_line_count == 0
    assert ruby.line_postprocessor("after\n") == "after\n"


def test_ruby_error_marker_is_translated_for_the_shared_reader():
    """The block's error marker becomes the canonical one the reader detects.

    subprocess_language.detect_end_of_execution() only knows the fixed strings,
    and handle_stream_output() strips the fixed end marker from output. Ruby
    must translate its per-block marker back before returning the line.
    """
    ruby = Ruby()
    ruby.preprocess_code("raise 'boom'")
    assert ruby.code_line_count > 0

    line = ruby.line_postprocessor(ruby._error_marker + "\n")

    assert line == "##execution_error##\n"
    assert ruby.code_line_count == 0
    assert ruby.detect_end_of_execution(line) is True


def test_ruby_line_postprocessor_skips_echoed_code():
    """Ruby line_postprocessor returns None while still within the submitted code line count."""
    ruby = Ruby()
    ruby.preprocess_code("puts 1")
    assert ruby.line_postprocessor("ignored") is None


def test_ruby_line_postprocessor_filters_nil():
    """Ruby line_postprocessor drops => nil result lines from output."""
    ruby = Ruby()
    ruby.code_line_count = 0
    assert ruby.line_postprocessor("=> nil") is None


def test_ruby_detect_active_line():
    """Ruby detect_active_line() parses ##active_lineN## markers and ignores plain text."""
    ruby = Ruby()
    assert ruby.detect_active_line("##active_line3##") == 3
    assert ruby.detect_active_line("hello") is None


def test_ruby_detect_end_of_execution():
    """Ruby detect_end_of_execution() recognizes end and error markers but not normal output."""
    ruby = Ruby()
    assert ruby.detect_end_of_execution("##end_of_execution##") is True
    assert ruby.detect_end_of_execution("##execution_error##") is True
    assert ruby.detect_end_of_execution("running") is False


def test_ruby_start_cmd_is_irb():
    """Ruby starts the irb REPL via ["irb"]."""
    assert Ruby().start_cmd == ["irb"]


def test_ruby_detect_end_of_execution_covers_error_marker():
    """The Ruby end detector treats both the end and error markers as completion.

    The rescue path emits ##execution_error## instead of ##end_of_execution##;
    the detector must accept either or errored scripts would never complete.
    """
    ruby = Ruby()
    assert ruby.detect_end_of_execution("##execution_error##") is True
    assert ruby.detect_end_of_execution("##end_of_execution##") is True


def test_ruby_line_postprocessor_consumes_exactly_the_code_line_count():
    """Each postprocess call decrements code_line_count by exactly one.

    The counter is set from the wrapped code and consumed one line per call; a
    -= 2 mutation would skip real output lines after the echoed code.
    """
    ruby = Ruby()
    ruby.preprocess_code("puts 1")
    start = ruby.code_line_count

    ruby.line_postprocessor("first")
    assert ruby.code_line_count == start - 1

    ruby.line_postprocessor("second")
    assert ruby.code_line_count == start - 2


def test_ruby_end_marker_survives_a_drifted_echo_window():
    """An exact end marker is passed through even while the skip window is open.

    irb echoes more lines than the wrapped block contains (banner, blanks,
    wrapped lines), so the counter can still be positive when the real end
    marker arrives. Skipping it leaves run() waiting forever for a completion
    signal that was already consumed, so the postprocessor must let exact
    control markers through and close the window.
    """
    ruby = Ruby()
    ruby.preprocess_code("1+1")
    assert ruby.code_line_count > 0

    line = ruby.line_postprocessor(ruby._end_marker + "\n")

    assert line == "##end_of_execution##\n"
    assert ruby.code_line_count == 0


def test_ruby_active_line_marker_survives_a_drifted_echo_window():
    """The block's active-line marker bypasses the skip window and resets it.

    The same drift that swallowed the end marker also dropped the highlight
    marker, so later runs showed no active line even while executing. Only the
    per-block marker (which carries the nonce) may do this; an exact canonical
    marker is an echoed user comment and must stay inside the window.
    """
    ruby = Ruby()
    processed = ruby.preprocess_code("1+1")
    assert ruby.code_line_count > 0

    marker = re.search(r"##active_line1_[0-9a-f]{8}##", processed).group(0)
    line = ruby.line_postprocessor(marker + "\n")

    assert line == "##active_line1##\n"
    assert ruby.code_line_count == 0


def test_ruby_echoed_active_line_comment_does_not_clear_the_echo_window():
    """An echoed active-line comment must not expose the next echoed marker.

    irb echoes comments before executing them. If a comment spelling
    ##active_line1## closed the window, the echoed ##end_of_execution##
    comment on the next line would reach the shared reader and report
    completion while the block was still running. Both comments must be
    consumed as echoes, and only the block's own marker may end it.
    """
    ruby = Ruby()
    processed = ruby.preprocess_code(
        '##active_line1##\n##end_of_execution##\nputs "after"'
    )
    assert ruby.code_line_count > 0

    for raw in processed.splitlines():
        assert ruby.line_postprocessor(raw + "\n") is None

    assert ruby.code_line_count == 0
    assert ruby.line_postprocessor("after\n") == "after\n"
    end_line = ruby.line_postprocessor(ruby._end_marker + "\n")
    assert end_line == "##end_of_execution##\n"


def test_ruby_echoed_marker_source_is_still_skipped():
    """The wrapper echo that spells the marker inside `puts` stays in the window.

    irb echoes the submitted source, including the `puts "marker"` wrapper line.
    That text is not an exact marker (the runtime line has no `puts` prefix),
    so treating it as completion would end the block while irb is still
    reading the input.
    """
    ruby = Ruby()
    ruby.preprocess_code("1+1")
    assert ruby.code_line_count > 0

    wrapper_line = f'    puts "{ruby._end_marker}\\n"\n'
    assert ruby.line_postprocessor(wrapper_line) is None
    assert ruby.line_postprocessor('    puts "##end_of_execution##\\n"\n') is None


def test_ruby_end_marker_after_print_output_is_translated():
    """A block end marker appended to newline-less output still completes.

    Ruby's `print` writes no trailing newline, so the ensure clause puts the
    block's marker on the same line as the output (`hello##end_of_execution_…`).
    An exact match missed it, and the shared reader only recognizes the
    canonical string, so run() waited on a completion signal that never came.
    The text before the marker is real output and must survive.
    """
    ruby = Ruby()
    ruby.preprocess_code('print "hello"')
    assert ruby.code_line_count > 0

    line = ruby.line_postprocessor("hello" + ruby._end_marker + "\n")

    assert line == "hello##end_of_execution##\n"
    assert ruby.code_line_count == 0
    assert ruby.detect_end_of_execution(line)


def test_ruby_error_marker_after_print_output_is_translated():
    """An exception raised after a newline-less print still reports the error.

    The error marker closes the printed line and the message follows on the
    next one. Without a suffix match the marker was swallowed by the echo
    window, so the reader never saw the completion/error marker.
    """
    ruby = Ruby()
    ruby.preprocess_code('print "x"; raise "boom"')
    assert ruby.code_line_count > 0

    line = ruby.line_postprocessor("x" + ruby._error_marker + "\n")

    assert line == "x##execution_error##\n"
    assert ruby.code_line_count == 0
    assert ruby.detect_end_of_execution(line)


def test_ruby_active_marker_after_print_output_is_translated():
    """An active-line marker appended to newline-less output still reports the line.

    `print "hello"` leaves the cursor on the line, so irb's marker for the next
    statement arrives as `hello##active_line2_NONCE##`. fullmatch() rejected
    that line, the raw nonce marker reached the shared reader, and
    detect_active_line() raised ValueError on the nonce suffix, aborting the
    stream. The printed text must survive ahead of the canonical marker.
    """
    ruby = Ruby()
    processed = ruby.preprocess_code('print "hello"\nputs "world"')
    marker = re.search(r"##active_line2_[0-9a-f]{8}##", processed).group(0)

    line = ruby.line_postprocessor("hello" + marker + "\n")

    assert line == "hello##active_line2##\n"
    assert ruby.code_line_count == 0
    assert ruby.detect_active_line(line) == 2


def test_ruby_nil_substring_is_real_output():
    """Only a bare nil result is filtered, not output that contains `nil`.

    The old substring test discarded real values such as "vanilla" and
    "nilpotent" because they contain the letters n-i-l.
    """
    ruby = Ruby()
    ruby.code_line_count = 0

    assert ruby.line_postprocessor("vanilla\n") == "vanilla\n"
    assert ruby.line_postprocessor("nilpotent\n") == "nilpotent\n"
    assert ruby.line_postprocessor("nil\n") is None
    assert ruby.line_postprocessor("=> nil\n") is None


def test_ruby_echo_window_counts_echoed_lines_not_split_segments():
    """The skip window spans exactly the lines irb echoes back.

    split("\n") yields a phantom trailing element for the wrapper's final
    newline, so the window stayed open one line past the echo. On the first
    block irb's session banner spent that extra line; on later blocks it ate
    the block's first real output line.
    """
    ruby = Ruby()
    processed = ruby.preprocess_code('puts "C"; puts "D"')

    assert ruby.code_line_count == len(processed.rstrip("\n").split("\n"))
    assert ruby.code_line_count == len(processed.splitlines())


def test_ruby_session_banner_does_not_consume_a_source_line():
    """irb's one-time banner is consumed without spending an echoed line.

    If the banner counted as an echoed line, the window would close before the
    last echoed source line and leak that line into the output as text.
    """
    ruby = Ruby()
    ruby.preprocess_code("1+1")
    start = ruby.code_line_count

    assert ruby.line_postprocessor("Switch to inspect mode.\n") is None
    assert ruby.code_line_count == start


def test_ruby_echo_window_closes_before_the_first_output_line():
    """After the echoed block, the first real output line is not swallowed.

    Feeds every echoed line (as irb prints them on a block after the banner)
    and then the first output line; the window must already be closed.
    """
    ruby = Ruby()
    processed = ruby.preprocess_code('puts "C"; puts "D"')
    for raw in processed.splitlines():
        assert ruby.line_postprocessor(raw + "\n") is None

    assert ruby.code_line_count == 0
    assert ruby.line_postprocessor("C\n") == "C\n"


def _started_ruby_or_skip():
    """Start an irb-backed Ruby session, skipping when the REPL is unavailable."""
    if shutil.which("irb") is None:
        pytest.skip("irb not installed; Ruby REPL coverage needs it")

    ruby = Ruby()
    try:
        ruby.start_process()
    except OSError as exc:
        # On Windows CI `shutil.which("irb")` can resolve a shim that Popen
        # still cannot launch; an absent REPL is a skip, not a failure.
        pytest.skip(f"irb could not be started: {exc}")
    return ruby


def _run_output(ruby, code):
    """Return only the program output chunks produced by one ruby.run() block."""
    return "".join(
        str(chunk["content"])
        for chunk in ruby.run(code)
        if chunk.get("format") == "output"
    )


def _output_lines(text):
    """Return the non-blank lines the user would see as program output."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def test_ruby_later_blocks_keep_their_first_output_line():
    """Live irb: a session's second block prints both output lines, exactly.

    The skipped line only appeared after the one-time banner was gone, so a
    single-block test cannot catch it. Asserting the exact lines also rejects
    echoed Ruby source that leaks into the output as text.
    """
    ruby = _started_ruby_or_skip()
    try:
        first = _run_output(ruby, 'puts "A"; puts "B"')
        second = _run_output(ruby, 'puts "C"; puts "D"')
    finally:
        ruby.terminate()

    assert _output_lines(first) == ["A", "B"]
    assert _output_lines(second) == ["C", "D"]


def test_ruby_live_comment_matching_the_marker_does_not_end_the_block():
    """Live irb: a comment spelling the historical marker does not finish early.

    irb echoes the comment before compiling it; only the block's own per-run
    marker may close the window, so the code after the comment still runs.
    """
    ruby = _started_ruby_or_skip()
    try:
        output = _run_output(ruby, '##end_of_execution##\nputs "after"')
    finally:
        ruby.terminate()

    assert _output_lines(output) == ["after"]


def test_ruby_live_marker_comments_do_not_end_the_block():
    """Live irb: both canonical marker comments together still run fully.

    An active-line comment used to close the echo window early, so the
    echoed completion-marker comment on the next line reported completion
    before irb ran the rest of the block.
    """
    ruby = _started_ruby_or_skip()
    try:
        output = _run_output(
            ruby, '##active_line1##\n##end_of_execution##\nputs "after"'
        )
    finally:
        ruby.terminate()

    assert _output_lines(output) == ["after"]


def test_ruby_live_print_without_newline_completes():
    """Live irb: a newline-less print ends the block and returns its output.

    The completion marker lands on the same line as the printed text, which is
    the shape the offline suffix tests model. This drives the real reader and
    run() loop. A join timeout makes a regression fail instead of hanging CI
    forever on a block that never reports completion.
    """
    ruby = _started_ruby_or_skip()
    chunks = []

    def collect():
        chunks.extend(ruby.run('print "hello"'))

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    thread.join(30)
    alive = thread.is_alive()
    try:
        assert not alive, "run() did not finish; the end marker was swallowed"
    finally:
        ruby.terminate()

    output = "".join(
        str(chunk["content"]) for chunk in chunks if chunk.get("format") == "output"
    )
    assert _output_lines(output) == ["hello"]


def test_ruby_live_print_then_puts_advances_the_active_line():
    """Live irb: `print` followed by `puts` reports line 2 and prints both.

    The second statement's marker shares the printed line, which is the shape
    the offline suffix test models. A regression would either raise ValueError
    inside the reader (raw nonce marker leaks) or lose the first line of
    output. The join timeout turns a hang into a failure instead of blocking CI.
    """
    ruby = _started_ruby_or_skip()
    chunks = []

    def collect():
        chunks.extend(ruby.run('print "hello"\nputs "world"'))

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    thread.join(30)
    alive = thread.is_alive()
    try:
        assert not alive, "run() did not finish; the active-line marker was lost"
    finally:
        ruby.terminate()

    active_lines = [
        chunk["content"] for chunk in chunks if chunk.get("format") == "active_line"
    ]
    assert 2 in active_lines
    output = "".join(
        str(chunk["content"]) for chunk in chunks if chunk.get("format") == "output"
    )
    assert _output_lines(output) == ["hello", "world"]
