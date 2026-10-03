import ast
import queue
import threading
import time
from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.terminal.languages.jupyter_language import (
    AddLinePrints,
    JupyterLanguage,
    add_active_line_prints,
    preprocess_python,
    string_to_python,
)


def _make_language():
    """Instantiate JupyterLanguage without running its kernel-starting __init__."""
    return JupyterLanguage.__new__(JupyterLanguage)


def test_detect_active_line_extracts_number_and_strips_marker():
    """detect_active_line() returns the active line number and removes the marker from the text."""
    lang = _make_language()
    assert lang.detect_active_line("print(1)\n##active_line2##\n") == ("print(1)\n", 2)


def test_detect_active_line_returns_none_when_no_marker():
    """detect_active_line() returns (line, None) when the line has no active-line marker."""
    lang = _make_language()
    assert lang.detect_active_line("just text") == ("just text", None)


def test_detect_active_line_non_numeric_marker_yields_zero():
    """detect_active_line() falls back to line 0 when the marker number is not an integer."""
    lang = _make_language()
    assert lang.detect_active_line("##active_lineabc##\n") == ("##active_lineabc##\n", 0)


def test_detect_active_line_strips_marker_from_trailing_content():
    """detect_active_line() returns the trailing text after the marker as the cleaned line."""
    lang = _make_language()
    assert lang.detect_active_line("##active_line5##\ncontent") == ("content", 5)


def test_preprocess_python_skips_injection_when_magic_present():
    """preprocess_python() skips active-line injection when any line is IPython magic, preserving code."""
    code = "%matplotlib inline\nx = 1"
    result = preprocess_python(code)
    assert result == "%matplotlib inline\nx = 1"


def test_add_active_line_prints_tracks_multiline_strings():
    """add_active_line_prints() keeps lines inside triple-quoted strings from being replaced with pass."""
    code = 's = """hello\nworld"""\nprint(s)'
    result = add_active_line_prints(code)
    assert "hello" in result
    assert "world" in result
    assert "##active_line" in result


def test_add_active_line_prints_replaces_comment_lines_with_pass():
    """add_active_line_prints() replaces comment-only lines with pass to preserve line numbering."""
    code = "# a comment\nx = 1"
    result = add_active_line_prints(code)
    assert "pass" in result
    assert "x = 1" in result


def test_add_line_prints_visit_processes_orelse_blocks():
    """AddLinePrints.visit() injects markers into the orelse block of an if statement."""
    code = "if True:\n    x = 1\nelse:\n    y = 2"
    tree = ast.parse(code)
    out = ast.unparse(AddLinePrints().visit(tree))
    assert "print('##active_line" in out
    assert "y = 2" in out


def test_add_line_prints_visit_processes_try_handlers_and_finalbody():
    """AddLinePrints.visit() injects markers into except handlers and the finally block of a try."""
    code = "try:\n    a = 1\nexcept ValueError:\n    b = 2\nfinally:\n    c = 3"
    tree = ast.parse(code)
    out = ast.unparse(AddLinePrints().visit(tree))
    assert "except ValueError" in out
    assert "finally" in out
    assert out.count("##active_line") >= 3


def test_add_line_prints_visit_handles_non_list_body():
    """AddLinePrints.visit() treats a non-list body as a single statement when processing."""
    transformer = AddLinePrints()
    # A single Assign statement is not wrapped in a list; process_body coerces it.
    body = ast.parse("x = 1").body[0]
    processed = transformer.process_body(body)
    assert len(processed) == 2
    assert isinstance(processed[0], ast.Expr)


def test_string_to_python_keeps_import_aliases():
    """string_to_python() preserves import aliases (import x as y) in the extracted function."""
    code = "import os as operating_system\n\ndef greet():\n    return 'hi'"
    functions = string_to_python(code)
    assert "import os as operating_system" in functions["greet"]


def test_string_to_python_skips_private_functions():
    """string_to_python() excludes functions whose names start with an underscore."""
    code = "def _helper():\n    pass\n\ndef pub():\n    return 1"
    functions = string_to_python(code)
    assert "_helper" not in functions
    assert "pub" in functions


def test_string_to_python_handles_docstring_less_function():
    """string_to_python() extracts functions without a docstring, using None as the placeholder."""
    code = "def add(a, b):\n    return a + b"
    functions = string_to_python(code)
    assert "def add():" in functions["add"]


def _scripted_language(msgs):
    """Build a JupyterLanguage instance whose kernel client replays scripted iopub messages.

    Instantiates via __new__ to skip the kernel-starting __init__, then wires a
    fake kernel client that returns each message in turn before raising queue.Empty.
    """
    lang = JupyterLanguage.__new__(JupyterLanguage)
    lang.finish_flag = False
    lang.last_output_time = time.time()
    lang.last_output_message_time = time.time()
    lang.computer = SimpleNamespace(
        interpreter=SimpleNamespace(stop_event=threading.Event())
    )

    class FakeIopub:
        def __init__(self, msgs):
            self.msgs = list(msgs)
            self.idx = 0

        def get_msg(self, timeout=0.05):
            if self.idx < len(self.msgs):
                msg = self.msgs[self.idx]
                self.idx += 1
                return msg
            raise queue.Empty()

    lang.kc = SimpleNamespace(
        iopub_channel=FakeIopub(msgs),
        input=lambda s: None,
        execute=lambda code: None,
        interrupt_kernel=lambda: None,
    )
    lang.km = SimpleNamespace(interrupt_kernel=lambda: None)
    return lang


def _iopub_msg(msg_type, **content):
    return {"header": {"msg_type": msg_type}, "msg_type": msg_type, "content": content}


def _drain_listener(lang):
    """Run _execute_code and collect everything the listener enqueued."""
    message_queue = queue.Queue()
    lang._execute_code("print(1)", message_queue)
    lang.listener_thread.join(timeout=5)
    assert not lang.listener_thread.is_alive(), "listener did not stop after status/idle"
    outputs = []
    while not message_queue.empty():
        outputs.append(message_queue.get())
    return outputs


def test_execute_code_dispatches_stream_output():
    """_execute_code() turns a stream iopub message into a console output chunk."""
    lang = _scripted_language(
        [_iopub_msg("stream", text="hello\n"), _iopub_msg("status", execution_state="idle")]
    )
    assert _drain_listener(lang) == [
        {"type": "console", "format": "output", "content": "hello\n"}
    ]


def test_execute_code_dispatches_stream_active_line():
    """_execute_code() emits an active_line chunk before the output when a marker is present."""
    lang = _scripted_language(
        [
            _iopub_msg("stream", text="##active_line3##\ncode"),
            _iopub_msg("status", execution_state="idle"),
        ]
    )
    assert _drain_listener(lang) == [
        {"type": "console", "format": "active_line", "content": 3},
        {"type": "console", "format": "output", "content": "code"},
    ]


def test_execute_code_dispatches_error_and_strips_ansi():
    """_execute_code() joins error tracebacks and strips ANSI escape codes before enqueueing."""
    lang = _scripted_language(
        [
            _iopub_msg("error", traceback=["Traceback...", "\x1b[31mValueError\x1b[0m"]),
            _iopub_msg("status", execution_state="idle"),
        ]
    )
    assert _drain_listener(lang) == [
        {"type": "console", "format": "output", "content": "Traceback...\nValueError"}
    ]


def test_execute_code_dispatches_png_image():
    """_execute_code() emits a base64.png image chunk for display_data with image/png."""
    lang = _scripted_language(
        [
            _iopub_msg("display_data", data={"image/png": "AAAA"}),
            _iopub_msg("status", execution_state="idle"),
        ]
    )
    assert _drain_listener(lang) == [
        {"type": "image", "format": "base64.png", "content": "AAAA"}
    ]


def test_execute_code_dispatches_jpeg_image():
    """_execute_code() emits a base64.jpeg image chunk for execute_result with image/jpeg."""
    lang = _scripted_language(
        [
            _iopub_msg("execute_result", data={"image/jpeg": "BBBB"}),
            _iopub_msg("status", execution_state="idle"),
        ]
    )
    assert _drain_listener(lang) == [
        {"type": "image", "format": "base64.jpeg", "content": "BBBB"}
    ]


def test_execute_code_dispatches_html():
    """_execute_code() emits a code/html chunk for display_data with text/html."""
    lang = _scripted_language(
        [
            _iopub_msg("display_data", data={"text/html": "<b>hi</b>"}),
            _iopub_msg("status", execution_state="idle"),
        ]
    )
    assert _drain_listener(lang) == [
        {"type": "code", "format": "html", "content": "<b>hi</b>"}
    ]


def test_execute_code_dispatches_text_plain():
    """_execute_code() emits a console output chunk for execute_result with text/plain."""
    lang = _scripted_language(
        [
            _iopub_msg("execute_result", data={"text/plain": "42"}),
            _iopub_msg("status", execution_state="idle"),
        ]
    )
    assert _drain_listener(lang) == [
        {"type": "console", "format": "output", "content": "42"}
    ]


def test_execute_code_dispatches_javascript():
    """_execute_code() emits a code/javascript chunk for display_data with application/javascript."""
    lang = _scripted_language(
        [
            _iopub_msg("display_data", data={"application/javascript": "console.log(1)"}),
            _iopub_msg("status", execution_state="idle"),
        ]
    )
    assert _drain_listener(lang) == [
        {"type": "code", "format": "javascript", "content": "console.log(1)"}
    ]


def test_execute_code_interrupts_when_finish_flag_set():
    """_execute_code() listener interrupts the kernel immediately when finish_flag is already set."""
    lang = _scripted_language([])
    lang.finish_flag = True
    interrupted = []
    lang.km.interrupt_kernel = lambda: interrupted.append(True)
    lang.kc.iopub_channel.get_msg = lambda timeout=0.05: (_ for _ in ()).throw(queue.Empty())
    lang._execute_code("x", SimpleNamespace(put=lambda *a: None))
    lang.listener_thread.join(timeout=5)
    assert interrupted == [True]


def test_capture_output_yields_queued_items_then_stops():
    """_capture_output() yields enqueued chunks and breaks once finish_flag is set and the queue empties."""
    lang = JupyterLanguage.__new__(JupyterLanguage)
    lang.finish_flag = True
    lang.listener_thread = threading.Thread
    lang.computer = SimpleNamespace(
        interpreter=SimpleNamespace(stop_event=threading.Event())
    )
    message_queue = queue.Queue()
    message_queue.put({"type": "console", "format": "output", "content": "one"})
    with mock.patch("interpreter.core.computer.terminal.languages.jupyter_language.time.sleep"):
        assert list(lang._capture_output(message_queue)) == [
            {"type": "console", "format": "output", "content": "one"}
        ]


def test_capture_output_breaks_on_stop_event():
    """_capture_output() sets finish_flag and stops early when the interpreter stop_event is set."""
    lang = JupyterLanguage.__new__(JupyterLanguage)
    stop = threading.Event()
    stop.set()
    lang.finish_flag = False
    lang.listener_thread = None
    lang.computer = SimpleNamespace(interpreter=SimpleNamespace(stop_event=stop))
    message_queue = queue.Queue()
    with mock.patch("interpreter.core.computer.terminal.languages.jupyter_language.time.sleep"):
        assert list(lang._capture_output(message_queue)) == []
    assert lang.finish_flag is True


def test_run_yields_captured_output():
    """run() preprocesses, executes, and forwards the captured output chunks to the caller."""
    lang = JupyterLanguage.__new__(JupyterLanguage)
    lang.finish_flag = False
    lang.kc = SimpleNamespace(is_alive=lambda: True)
    lang.computer = SimpleNamespace(
        interpreter=SimpleNamespace(stop_event=threading.Event())
    )
    lang.preprocess_code = mock.Mock(return_value="processed")
    lang._execute_code = mock.Mock()
    lang._capture_output = mock.Mock(
        return_value=iter(
            [{"type": "console", "format": "output", "content": "ran"}]
        )
    )
    assert list(lang.run("print(1)")) == [
        {"type": "console", "format": "output", "content": "ran"}
    ]
    lang.preprocess_code.assert_called_once_with("print(1)")
    lang._execute_code.assert_called_once_with("processed", mock.ANY)


def test_run_yields_error_content_on_execution_failure():
    """run() converts an exception from _execute_code into a console output chunk."""
    lang = JupyterLanguage.__new__(JupyterLanguage)
    lang.finish_flag = False
    lang.kc = SimpleNamespace(is_alive=lambda: True)
    lang.computer = SimpleNamespace(
        interpreter=SimpleNamespace(stop_event=threading.Event())
    )
    lang.preprocess_code = lambda code: code
    lang._capture_output = lambda mq: iter([])

    def boom(code, mq):
        raise RuntimeError("kaboom")

    lang._execute_code = boom
    outputs = list(lang.run("x"))
    assert len(outputs) == 1
    assert outputs[0]["type"] == "console"
    assert outputs[0]["format"] == "output"
    assert "RuntimeError: kaboom" in outputs[0]["content"]


def test_stop_sets_finish_flag():
    """stop() flags the language so the listener thread halts at the next check."""
    lang = JupyterLanguage.__new__(JupyterLanguage)
    lang.finish_flag = False
    lang.stop()
    assert lang.finish_flag is True


def test_terminate_stops_channels_and_shuts_down_kernel():
    """terminate() closes the kernel client channels and shuts down the kernel."""
    lang = JupyterLanguage.__new__(JupyterLanguage)
    calls = []
    lang.kc = SimpleNamespace(stop_channels=lambda: calls.append("stop_channels"))
    lang.km = SimpleNamespace(shutdown_kernel=lambda: calls.append("shutdown_kernel"))
    lang.terminate()
    assert calls == ["stop_channels", "shutdown_kernel"]


def test_preprocess_code_delegates_to_preprocess_python():
    """preprocess_code() forwards to preprocess_python() with the same env-flag behavior."""
    lang = JupyterLanguage.__new__(JupyterLanguage)
    with mock.patch.dict(
        "os.environ", {"INTERPRETER_ACTIVE_LINE_DETECTION": "false"}
    ):
        assert lang.preprocess_code("x = 1\n\n") == "x = 1"


def _iopub_msg_from(msg_type, parent_msg_id, **content):
    """Build an iopub message tagged with the execute_request that caused it."""
    return {
        "header": {"msg_type": msg_type},
        "msg_type": msg_type,
        "parent_header": {"msg_id": parent_msg_id},
        "content": content,
    }


def test_execute_code_ignores_a_previous_executions_messages():
    """Leftover output from an earlier execute_request is not read as this one's.

    interrupt_kernel() returns before the kernel has finished emitting, and
    stop() abandons the listener without draining, so the next run() can find
    the previous execution's stream and status messages still queued. Reading
    the stale status/idle ended this execution immediately, so its real output
    stayed in the channel and surfaced under the command after it — output
    arriving one command late, and commands that appeared never to run.
    """
    lang = _scripted_language(
        [
            _iopub_msg_from("stream", "old-request", name="stdout", text="STALE\n"),
            _iopub_msg_from("status", "old-request", execution_state="idle"),
            _iopub_msg_from("stream", "this-request", name="stdout", text="MINE\n"),
            _iopub_msg_from("status", "this-request", execution_state="idle"),
        ]
    )
    lang.kc.execute = lambda code: "this-request"

    outputs = _drain_listener(lang)

    texts = [o.get("content") for o in outputs if o.get("type") == "console"]
    assert "MINE\n" in texts
    assert "STALE\n" not in texts


def test_execute_code_records_the_execution_id_before_listening():
    """The execute_request is sent before the listener starts, so its id is known.

    The listener filters on parent_header against this id; if it started first
    it would race the assignment and could read stale traffic as its own.
    """
    lang = _scripted_language(
        [_iopub_msg_from("status", "req-1", execution_state="idle")]
    )
    lang.kc.execute = lambda code: "req-1"

    _drain_listener(lang)

    assert lang._execution_id == "req-1"


def test_execute_code_still_accepts_messages_without_a_parent_header():
    """Messages carrying no parent_header are kept rather than discarded.

    Some kernel messages arrive unparented; dropping them would lose real
    output, so the filter only rejects a parent that is positively someone
    else's.
    """
    lang = _scripted_language(
        [
            _iopub_msg("stream", name="stdout", text="UNPARENTED\n"),
            _iopub_msg_from("status", "this-request", execution_state="idle"),
        ]
    )
    lang.kc.execute = lambda code: "this-request"

    outputs = _drain_listener(lang)

    assert "UNPARENTED\n" in [o.get("content") for o in outputs if o.get("type") == "console"]
