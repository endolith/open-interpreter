import ast
import threading
from types import SimpleNamespace
from unittest import mock

import interpreter.core.computer.terminal.languages.jupyter_language as jl
from interpreter.core.computer.terminal.languages.jupyter_language import (
    AddLinePrints,
    add_active_line_prints,
    preprocess_python,
    string_to_python,
    wrap_in_try_except,
)
from interpreter.core.computer.terminal.languages.python import Python


def test_preprocess_python_adds_active_line_markers():
    """preprocess_python() injects ##active_line markers into executable Python lines."""
    code = "x = 1\nprint(x)"
    result = preprocess_python(code)
    assert "##active_line" in result


def test_preprocess_python_skips_magic_lines():
    """preprocess_python() leaves IPython magic lines without active-line markers."""
    code = "%matplotlib inline\nx = 1"
    result = preprocess_python(code)
    assert "##active_line" not in result


def test_preprocess_python_respects_active_line_detection_env(monkeypatch):
    """INTERPRETER_ACTIVE_LINE_DETECTION=false disables marker injection."""
    monkeypatch.setenv("INTERPRETER_ACTIVE_LINE_DETECTION", "false")
    code = "x = 1"
    result = preprocess_python(code)
    assert "##active_line" not in result


def test_preprocess_python_strips_blank_lines(monkeypatch):
    """preprocess_python() removes blank lines when active-line detection is disabled."""
    monkeypatch.setenv("INTERPRETER_ACTIVE_LINE_DETECTION", "false")
    code = "x = 1\n\n\ny = 2"
    result = preprocess_python(code)
    assert result == "x = 1\ny = 2"


def test_add_active_line_prints_inserts_before_executable_lines():
    """add_active_line_prints() adds markers before each executable Python statement."""
    code = "a = 1\nprint(a)"
    result = add_active_line_prints(code)
    assert result.count("##active_line") >= 2


def test_add_active_line_prints_handles_comments():
    """add_active_line_prints() skips comment-only lines but still processes executable code."""
    code = "# comment\nx = 1"
    result = add_active_line_prints(code)
    assert "x = 1" in result


def test_string_to_python_extracts_public_functions():
    """string_to_python() returns public function definitions with their imports, excluding private names."""
    code = """
import os

def hello():
    \"\"\"Say hi\"\"\"
    return 1

def _private():
    pass
"""
    functions = string_to_python(code)
    assert "hello" in functions
    assert "_private" not in functions
    assert "def hello():" in functions["hello"]
    assert "import os" in functions["hello"]


def test_wrap_in_try_except_wraps_code():
    """wrap_in_try_except() wraps user code in try/except with traceback printing."""
    code = "x = 1"
    result = wrap_in_try_except(code)
    assert "try:" in result
    assert "traceback.print_exc" in result


def test_add_line_prints_transformer_inserts_prints():
    """AddLinePrints AST transformer injects ##active_line print statements into parsed code."""
    tree = ast.parse("x = 1")
    transformer = AddLinePrints()
    new_tree = transformer.visit(tree)
    unparsed = ast.unparse(new_tree)
    assert "##active_line" in unparsed


def test_kernel_is_alive_uses_the_process_not_the_client():
    """The client can stay up after the process exits; liveness must use the manager."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False
    assert lang._kernel_is_alive() is False

    lang.km.is_alive.return_value = True
    assert lang._kernel_is_alive() is True


def test_kernel_is_alive_assumes_life_without_a_manager():
    """Non-Jupyter setups have no manager; they must not be mistaken for a dead kernel."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.km = None
    assert lang._kernel_is_alive() is True


def test_kernel_is_alive_assumes_life_when_the_check_errors():
    """A manager that cannot answer must not trigger a destructive restart."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.km = mock.Mock()
    lang.km.is_alive.side_effect = RuntimeError("broken")
    assert lang._kernel_is_alive() is True


def test_ensure_kernel_restarts_a_dead_kernel_with_its_setup():
    """A dead kernel is recreated by re-running __init__, so OI's setup comes back."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.computer = object()
    lang.finish_flag = False
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False
    lang.kc = mock.Mock()

    restarted = []

    def fake_init(_self, computer):
        restarted.append(computer)

    with mock.patch.object(jl.JupyterLanguage, "__init__", fake_init):
        lang._ensure_kernel()

    assert lang.finish_flag is True
    lang.kc.stop_channels.assert_called_once_with()
    lang.km.shutdown_kernel.assert_called_once_with(now=True)
    assert restarted == [lang.computer]


def test_run_reports_a_kernel_that_cannot_be_restarted():
    """A failed restart is reported instead of running code against a dead kernel."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False
    lang.finish_flag = False

    with mock.patch.object(jl.JupyterLanguage, "_ensure_kernel", lambda self: None):
        chunks = list(lang.run("print(1)"))

    assert chunks == [
        {
            "type": "console",
            "format": "output",
            "content": "The Python kernel could not be restarted.",
        }
    ]


def test_wait_for_client_gives_up_after_the_timeout():
    """A client that never comes up must not block the run forever."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.kc = mock.Mock()
    lang.kc.is_alive.return_value = False

    clock = iter([0.0, 5.0, 10.0, 99.0])
    with mock.patch.object(jl.time, "time", lambda: next(clock)), mock.patch.object(
        jl.time, "sleep"
    ) as sleep:
        lang._wait_for_client()

    assert sleep.call_count == 1


def test_wait_for_client_returns_at_once_when_the_kernel_process_dies():
    """A replacement kernel that died during startup cannot bring its client up.

    Idling until the deadline would turn a failed restart into a pause of the
    whole terminal, so the dead process must end the wait immediately.
    """
    lang = object.__new__(jl.JupyterLanguage)
    lang.kc = mock.Mock()
    lang.kc.is_alive.return_value = False
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False

    with mock.patch.object(jl.time, "sleep") as sleep:
        lang._wait_for_client()

    sleep.assert_not_called()


def test_ensure_kernel_fences_and_joins_a_stale_listener_before_init():
    """A stale listener must be fenced off and joined before the client is replaced.

    Its generation is bumped so it stops on its own if it is blocked in the
    input-patience model call and outlives the bounded join; otherwise it could
    send input to, or stop, the recovered run.
    """
    lang = object.__new__(jl.JupyterLanguage)
    lang.computer = object()
    lang.finish_flag = False
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False
    lang.kc = mock.Mock()
    listener = mock.Mock()
    listener.is_alive.return_value = True
    lang.listener_thread = listener
    lang._listener_generation = 7

    restarted = []

    with mock.patch.object(
        jl.JupyterLanguage, "__init__", lambda _self, computer: restarted.append(computer)
    ):
        lang._ensure_kernel()

    listener.join.assert_called_once_with(timeout=5)
    assert lang._listener_generation == 8
    assert restarted == [lang.computer]


def test_run_reports_a_restart_that_raises_instead_of_leaking_the_traceback():
    """A restart that fails midway must be reported as an error chunk.

    __init__ can raise (start_kernel, channel setup, ...). Escaping as an
    exception would crash the terminal loop, and leaving a half-built kernel
    behind would make the next run fail again for a second, different reason.
    """
    lang = object.__new__(jl.JupyterLanguage)
    lang.km = mock.Mock()
    lang.kc = mock.Mock()

    with mock.patch.object(
        jl.JupyterLanguage, "_ensure_kernel", side_effect=RuntimeError("no kernel")
    ), mock.patch.object(jl.JupyterLanguage, "_stop_kernel") as stop:
        chunks = list(lang.run("print(1)"))

    stop.assert_called_once_with()
    assert len(chunks) == 1
    assert chunks[0] == {
        "type": "console",
        "format": "output",
        "content": chunks[0]["content"],
    }
    assert "RuntimeError" in chunks[0]["content"]
    assert "no kernel" in chunks[0]["content"]


def test_ensure_kernel_tolerates_failing_channel_and_shutdown_cleanup():
    """Cleanup errors must not stop the restart; the kernel still comes back."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.computer = object()
    lang.finish_flag = False
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False
    lang.km.shutdown_kernel.side_effect = RuntimeError("gone")
    lang.kc = mock.Mock()
    lang.kc.stop_channels.side_effect = RuntimeError("gone")

    restarted = []

    with mock.patch.object(
        jl.JupyterLanguage, "__init__", lambda _self, computer: restarted.append(computer)
    ):
        lang._ensure_kernel()

    assert restarted == [lang.computer]


def test_ensure_kernel_skips_a_missing_client():
    """A kernel manager without a client still restarts cleanly."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.computer = object()
    lang.finish_flag = False
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False
    lang.kc = None

    with mock.patch.object(jl.JupyterLanguage, "__init__", lambda _self, computer: None):
        lang._ensure_kernel()

    lang.km.shutdown_kernel.assert_called_once_with(now=True)


def test_capture_output_ends_when_the_kernel_dies():
    """A dead kernel drains its output, reports the exit, and ends the run."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.finish_flag = False
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False
    lang.computer = SimpleNamespace(interpreter=SimpleNamespace())
    lang.listener_thread = mock.Mock()
    lang.listener_thread.is_alive.return_value = True

    import queue as queue_module
    message_queue = queue_module.Queue()
    message_queue.put({"type": "console", "format": "output", "content": "first"})
    message_queue.put({"type": "console", "format": "output", "content": "second"})

    chunks = list(lang._capture_output(message_queue))

    assert chunks == [
        {"type": "console", "format": "output", "content": "first"},
        {"type": "console", "format": "output", "content": "second"},
        {
            "type": "console",
            "format": "output",
            "content": jl._KERNEL_EXIT_NOTICE,
        },
    ]
    assert lang.finish_flag is True


def test_capture_output_drains_output_queued_as_the_kernel_died():
    """Output that lands between the empty check and the liveness check is not lost."""
    import queue as queue_module

    class RacyQueue:
        def __init__(self, item):
            self.item = item

        def get(self, timeout=None):
            if timeout is not None:
                raise queue_module.Empty
            item, self.item = self.item, None
            return item

        def empty(self):
            return self.item is None

    lang = object.__new__(jl.JupyterLanguage)
    lang.finish_flag = False
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False
    lang.computer = SimpleNamespace(interpreter=SimpleNamespace(stop_event=threading.Event()))
    lang.listener_thread = mock.Mock()
    lang.listener_thread.is_alive.return_value = True

    pending = {"type": "console", "format": "output", "content": "late"}
    chunks = list(lang._capture_output(RacyQueue(pending)))

    assert chunks == [
        pending,
        {
            "type": "console",
            "format": "output",
            "content": jl._KERNEL_EXIT_NOTICE,
        },
    ]
    assert lang.finish_flag is True


def test_capture_output_ends_when_the_kernel_is_dead_and_nothing_is_queued():
    """With no queued output the dead-kernel branch must report the exit and stop the run."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.finish_flag = False
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False
    lang.computer = SimpleNamespace(interpreter=SimpleNamespace(stop_event=threading.Event()))
    lang.listener_thread = mock.Mock()
    lang.listener_thread.is_alive.return_value = True

    import queue as queue_module

    chunks = list(lang._capture_output(queue_module.Queue()))

    assert chunks == [
        {
            "type": "console",
            "format": "output",
            "content": jl._KERNEL_EXIT_NOTICE,
        }
    ]
    assert lang.finish_flag is True


def test_capture_output_reports_the_kernel_exit_only_once():
    """A notice the listener already queued must not be repeated by the drain."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.finish_flag = False
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = False
    lang.computer = SimpleNamespace(interpreter=SimpleNamespace(stop_event=threading.Event()))
    lang.listener_thread = mock.Mock()
    lang.listener_thread.is_alive.return_value = True

    import queue as queue_module

    notice = {
        "type": "console",
        "format": "output",
        "content": jl._KERNEL_EXIT_NOTICE,
    }
    message_queue = queue_module.Queue()
    message_queue.put(notice)

    chunks = list(lang._capture_output(message_queue))

    assert chunks == [notice]
    assert lang.finish_flag is True


def test_capture_output_interrupts_when_the_listener_thread_died():
    """A dead listener must interrupt the kernel and report why the run ended."""
    lang = object.__new__(jl.JupyterLanguage)
    lang.finish_flag = False
    lang.km = mock.Mock()
    lang.km.is_alive.return_value = True
    lang.computer = SimpleNamespace(interpreter=SimpleNamespace())
    lang.listener_thread = mock.Mock()
    lang.listener_thread.is_alive.return_value = False

    import queue as queue_module
    chunks = list(lang._capture_output(queue_module.Queue()))

    assert lang.km.interrupt_kernel.call_count == 1
    assert chunks == [
        {
            "type": "console",
            "format": "output",
            "content": (
                "The Python kernel stopped sending output; "
                "the run was interrupted."
            ),
        }
    ]
    assert lang.finish_flag is True


def _collect(generator, timeout=30):
    """Consume a generator on a thread, failing instead of hanging when it stalls."""
    result = {}

    def worker():
        try:
            result["chunks"] = list(generator)
        except Exception as exc:  # pragma: no cover - surfaced by the assertion
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "run() hung instead of returning"
    assert "error" not in result, result.get("error")
    return result["chunks"]


def test_report_kernel_exit_drains_sent_output_before_the_notice():
    """Output already delivered to the channel must precede the exit notice.

    A hard exit can land between the listener's last read and its liveness
    check, leaving the message that explains the exit unread. Reporting the
    notice first would show the user a dead kernel with no output, so the
    listener drains its own channel before queueing the notice.
    """
    import queue as queue_module

    lang = object.__new__(jl.JupyterLanguage)
    lang.finish_flag = False
    lang.last_output_time = 0.0
    lang.detect_active_line = mock.Mock(return_value=("before exit\n", None))

    stale = {
        "header": {"msg_type": "stream"},
        "msg_type": "stream",
        "parent_header": {"msg_id": "an-earlier-request"},
        "content": {"text": "stale\n"},
    }
    fresh = {
        "header": {"msg_type": "stream"},
        "msg_type": "stream",
        "parent_header": {"msg_id": "this-request"},
        "content": {"text": "before exit\n"},
    }
    channel = mock.Mock()
    channel.get_msg.side_effect = [stale, fresh, queue_module.Empty()]
    kc = SimpleNamespace(iopub_channel=channel)

    message_queue = queue_module.Queue()
    lang._report_kernel_exit(kc, "this-request", message_queue)

    chunks = []
    while not message_queue.empty():
        chunks.append(message_queue.get())

    assert chunks == [
        {"type": "console", "format": "output", "content": "before exit\n"},
        {
            "type": "console",
            "format": "output",
            "content": jl._KERNEL_EXIT_NOTICE,
        },
    ]
    assert lang.finish_flag is True
    assert lang.last_output_time > 0.0


def test_report_kernel_exit_queues_the_notice_when_the_channel_is_closed():
    """A channel that is already closed must not swallow the exit notice.

    The drain is an opportunity to recover sent output, not a prerequisite for
    reporting the exit, so a read that raises still ends with the notice.
    """
    import queue as queue_module

    lang = object.__new__(jl.JupyterLanguage)
    lang.finish_flag = False
    channel = mock.Mock()
    channel.get_msg.side_effect = RuntimeError("channel closed")
    kc = SimpleNamespace(iopub_channel=channel)

    message_queue = queue_module.Queue()
    lang._report_kernel_exit(kc, "this-request", message_queue)

    assert [message_queue.get()] == [
        {
            "type": "console",
            "format": "output",
            "content": jl._KERNEL_EXIT_NOTICE,
        }
    ]
    assert lang.finish_flag is True


def test_python_recovers_after_the_kernel_hard_exits():
    """os._exit(0) must end the run, and the next run must restart the kernel.

    A regression here hangs the REPL: the kernel is dead, the client channels
    stay open, and the run waits forever for an idle message that cannot arrive.
    """

    computer = SimpleNamespace(
        interpreter=SimpleNamespace(
            stop_event=threading.Event(),
            messages=[],
            verbose=False,
            debug=False,
            max_output=2000,
            plain_text_display=True,
        )
    )
    language = Python(computer)
    try:
        _collect(language.run("print(1)"))
        killed = _collect(
            language.run("import os, sys; sys.stdout.flush(); os._exit(0)")
        )
        notices = [
            chunk
            for chunk in killed
            if "kernel exited" in str(chunk.get("content", ""))
        ]
        assert len(notices) == 1
        revived = _collect(language.run("print(2)"))
        assert any("2" in str(chunk.get("content", "")) for chunk in revived)
    finally:
        language.terminate()
