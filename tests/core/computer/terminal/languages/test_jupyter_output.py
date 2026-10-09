"""Tests for how Jupyter iopub messages become interpreter output chunks.

This is the mapping every Python snippet's output travels through: whatever the
kernel emits on iopub is translated into the console/image/code chunks the rest
of the interpreter renders. Getting a mapping wrong loses output silently, since
nothing downstream knows a chunk was never produced.

Driven with a scripted stub kernel rather than a real one, so each branch is
reachable directly instead of by writing code that happens to produce that
output shape.
"""

import queue
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.computer.terminal.languages.jupyter_language import JupyterLanguage


def _msg(msg_type, content):
    """An iopub message with both the nested and flat keys the listener reads."""
    return {
        "msg_type": msg_type,
        "header": {"msg_type": msg_type},
        "content": content,
    }


def _run_listener(lang, messages):
    """Feed `messages` to the listener, then let it stop. Returns what it queued."""
    out = queue.Queue()
    pending = list(messages)

    def get_msg(timeout=None):
        if pending:
            return pending.pop(0)
        lang.finish_flag = True
        raise queue.Empty

    lang.kc.iopub_channel.get_msg = get_msg
    lang._execute_code("code", out)
    lang.listener_thread.join(timeout=10)
    assert not lang.listener_thread.is_alive(), "listener did not finish"

    drained = []
    while True:
        try:
            drained.append(out.get_nowait())
        except queue.Empty:
            return drained


def _language(*, stop_event_set=False):
    """A JupyterLanguage with a stub kernel and an interpreter double."""
    lang = JupyterLanguage.__new__(JupyterLanguage)
    lang.computer = SimpleNamespace(
        interpreter=SimpleNamespace(
            stop_event=SimpleNamespace(is_set=lambda: stop_event_set),
        )
    )
    lang.km = SimpleNamespace(interrupt_kernel=mock.Mock())
    lang.kc = SimpleNamespace(execute=mock.Mock(), input=mock.Mock(), iopub_channel=SimpleNamespace())
    lang.finish_flag = False
    # The listener consults both output clocks on every iteration, before touching
    # the channel. Without them the patience check raises AttributeError, which the
    # surrounding `except Exception` swallows and retries, so every message below
    # would be silently dropped instead of dispatched.
    lang.last_output_time = time.time()
    lang.last_output_message_time = time.time()
    return lang


def test_stream_output_becomes_a_console_output_chunk():
    """A `stream` message is the ordinary case: stdout arrives as output."""
    lang = _language()
    queued = _run_listener(lang, [_msg("stream", {"text": "hello\n"})])

    assert queued == [{"type": "console", "format": "output", "content": "hello\n"}]


def test_active_line_marker_is_split_out_from_stream_text():
    """`##active_line3##` becomes its own chunk, and is stripped from the text.

    `preprocess_code` injects these markers so the UI can highlight the running
    line. If they were not stripped they would appear verbatim in the user's
    output, so both halves are checked: the marker is reported, and it does not
    leak into the text.
    """
    lang = _language()
    queued = _run_listener(
        lang, [_msg("stream", {"text": "##active_line3##\nprint(1)\n"})]
    )

    assert queued[0] == {"type": "console", "format": "active_line", "content": 3}
    assert queued[1] == {"type": "console", "format": "output", "content": "print(1)\n"}
    assert "##active_line" not in queued[1]["content"], "marker must not leak into output"


def test_a_malformed_active_line_marker_is_dropped_rather_than_reported():
    """An unparseable marker yields no active_line chunk.

    `detect_active_line` falls back to `0` when the digits do not parse, and the
    listener branches on `if active_line:`, so 0 is the sentinel for "marker
    present but unreadable" and is deliberately not reported.

    This looked like it might be a dropped-line-zero bug — 0 is falsy, so the
    same branch would swallow a genuine line 0. It is not: `add_active_line_prints`
    injects `sub_node.lineno`, and AST line numbers are 1-based, so 0 cannot
    occur. Pinned as the sentinel it is, so the two meanings do not get confused.
    """
    lang = _language()
    text = "##active_lineXYZ##\nprint(1)\n"
    queued = _run_listener(lang, [_msg("stream", {"text": text})])

    assert [c["format"] for c in queued] == ["output"]
    # The text is passed through unchanged, marker included. Stripping is
    # digit-only (`##active_line\d+##`), so a non-numeric marker is not the
    # marker's format at all -- it is literal user text, and eating it would
    # delete something the user actually printed.
    assert queued[0]["content"] == text


def test_markers_only_ever_name_lines_one_or_higher():
    """Real markers carry 1-based AST line numbers, so the sentinel is unambiguous."""
    lang = _language()
    queued = _run_listener(
        lang, [_msg("stream", {"text": "##active_line1##\nfirst\n"})]
    )

    assert queued[0] == {"type": "console", "format": "active_line", "content": 1}


def test_an_error_traceback_is_joined_and_stripped_of_ansi_codes():
    """A kernel error is reported as joined traceback lines, without colour codes.

    Jupyter sends the traceback as a list of ANSI-coloured strings. Left in, the
    escape sequences render as garbage in a terminal and get copied into
    conversation history when the user quotes the error back.
    """
    lang = _language()
    traceback = ["\x1b[0;31mTraceback\x1b[0m", "\x1b[0;31mValueError: boom\x1b[0m"]
    queued = _run_listener(lang, [_msg("error", {"traceback": traceback})])

    assert len(queued) == 1
    content = queued[0]["content"]
    assert queued[0]["format"] == "output"
    assert "\x1b" not in content, "ANSI escapes must be removed"
    assert "Traceback" in content and "ValueError: boom" in content
    assert "\n" in content, "traceback entries are joined, not concatenated"


def test_an_idle_kernel_ends_the_listen():
    """The `status`/`idle` message is how a finished execution terminates.

    Without it the listener spins until the patience timer notices, so a
    completed snippet would appear to hang for the full window.
    """
    lang = _language()
    queued = _run_listener(
        lang,
        [
            _msg("stream", {"text": "done\n"}),
            _msg("status", {"execution_state": "idle"}),
            _msg("stream", {"text": "never seen\n"}),
        ],
    )

    assert [c["content"] for c in queued] == ["done\n"]
    assert lang.finish_flag is True


def test_a_busy_status_does_not_end_the_listen():
    """A `status` of `busy` is progress, not completion, so the listener continues."""
    lang = _language()
    queued = _run_listener(
        lang,
        [
            _msg("status", {"execution_state": "busy"}),
            _msg("stream", {"text": "still going\n"}),
        ],
    )

    # Proof the listener continued is that the message *after* the busy status was
    # dispatched, not the final finish_flag: this harness sets finish_flag itself
    # once the scripted messages run out.
    assert [c["content"] for c in queued] == ["still going\n"]


@pytest.mark.parametrize(
    "mime,expected",
    [
        (
            "image/png",
            {"type": "image", "format": "base64.png", "content": "PNGDATA"},
        ),
        (
            "image/jpeg",
            {"type": "image", "format": "base64.jpeg", "content": "JPEGDATA"},
        ),
        (
            "text/html",
            {"type": "code", "format": "html", "content": "<b>hi</b>"},
        ),
        (
            "text/plain",
            {"type": "console", "format": "output", "content": "plain"},
        ),
        (
            "application/javascript",
            {"type": "code", "format": "javascript", "content": "alert(1)"},
        ),
    ],
)
def test_display_data_maps_mime_types_to_chunks(mime, expected):
    """Each supported mime type becomes the chunk shape its consumer expects.

    The format strings are load-bearing: `base64.png` is what later code splits
    on to tell an inline image from a file path, and `html`/`javascript` mark the
    chunk as code for the renderer.
    """
    lang = _language()
    data = {
        "image/png": "PNGDATA",
        "image/jpeg": "JPEGDATA",
        "text/html": "<b>hi</b>",
        "text/plain": "plain",
        "application/javascript": "alert(1)",
    }
    queued = _run_listener(lang, [_msg("display_data", {"data": {mime: data[mime]}})])

    assert queued == [expected]


def test_execute_result_is_handled_like_display_data():
    """A value returned by an expression is mapped the same way.

    Jupyter sends `execute_result` for the value of a bare expression, which is
    how `2 + 2` or a bare DataFrame produces output — so a mapping that covered
    only `display_data` would drop the output of any expression-statement.
    """
    lang = _language()
    queued = _run_listener(
        lang, [_msg("execute_result", {"data": {"text/plain": "4"}})]
    )

    assert queued == [{"type": "console", "format": "output", "content": "4"}]


def test_image_precedence_puts_png_before_others():
    """When a result carries several representations, the image wins.

    matplotlib figures arrive with both `image/png` and `text/plain`; taking the
    image is the point, since the text form is just a placeholder string.
    """
    lang = _language()
    queued = _run_listener(
        lang,
        [
            _msg(
                "display_data",
                {"data": {"text/plain": "<Figure size 640x480>", "image/png": "PNGDATA"}},
            )
        ],
    )

    assert queued == [{"type": "image", "format": "base64.png", "content": "PNGDATA"}]


def test_an_unsupported_mime_type_produces_nothing():
    """Characterization: a mime type outside the list is dropped with no chunk.

    The `elif` chain has no fallback, so anything it does not name disappears
    without a warning — `application/json`, `image/svg+xml` and `text/markdown`
    all land here. Recorded because the drop is invisible: the snippet appears to
    produce no output and there is nothing in the transcript explaining why.
    """
    lang = _language()
    queued = _run_listener(
        lang, [_msg("display_data", {"data": {"application/json": '{"a": 1}'}})]
    )

    assert queued == []


def test_a_stop_event_interrupts_the_kernel():
    """A set `stop_event` interrupts the kernel and ends the listener.

    This is the async cancellation path: the listener polls the interpreter's
    stop event and, when it is set, interrupts rather than waiting for the code to
    finish on its own.
    """
    lang = _language(stop_event_set=True)
    _run_listener(lang, [])

    lang.km.interrupt_kernel.assert_called_once()
    assert lang.finish_flag is True


def test_the_finish_flag_also_interrupts_the_kernel():
    """Once `finish_flag` is set the listener interrupts rather than waiting.

    Distinct from the stop_event path: this is OI's own `stop()` setting the flag,
    which is how an interrupt request arrives from inside `stop()`. Both must
    interrupt, or a stopped snippet keeps running to completion.
    """
    lang = _language()
    lang.finish_flag = True
    _run_listener(lang, [])

    lang.km.interrupt_kernel.assert_called_once()
