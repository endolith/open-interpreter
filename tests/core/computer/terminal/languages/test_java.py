from unittest import mock

from interpreter.core.computer.terminal.languages.java import Java, preprocess_java

_VALID_CLASS = "class Hello { public static void main(String[] args) {} }"


def test_preprocess_java_adds_markers():
    """preprocess_java() injects active-line and end-of-execution markers into Java code."""
    code = preprocess_java("System.out.println(1);")
    assert "##active_line1##" in code
    assert "##end_of_execution##" in code


def test_java_run_without_class_yields_error():
    """Java.run() reports missing class definition when source has no class."""
    java = Java()
    chunks = list(java.run("not a class"))
    assert "No class definition found" in chunks[0]["content"]


def test_java_run_compilation_error_cleans_up_java_file(tmp_path, monkeypatch):
    """Java.run() removes the .java source file after a compilation failure."""
    monkeypatch.chdir(tmp_path)
    java = Java()

    compile_proc = mock.Mock()
    compile_proc.communicate.return_value = ("", "syntax error")
    compile_proc.returncode = 1

    with mock.patch(
        "interpreter.core.computer.terminal.languages.java.subprocess.Popen",
        return_value=compile_proc,
    ):
        chunks = list(java.run(_VALID_CLASS))

    assert "Compilation Error" in chunks[0]["content"]
    assert not (tmp_path / "Hello.java").exists()


def test_java_run_finally_removes_java_and_class_files(tmp_path, monkeypatch):
    """Java.run() finally block removes both .java and .class files after compile errors."""
    monkeypatch.chdir(tmp_path)
    java = Java()

    compile_proc = mock.Mock()
    compile_proc.returncode = 1

    def fail_compile_and_touch_class():
        (tmp_path / "Hello.class").write_bytes(b"fake")
        return ("", "syntax error")

    compile_proc.communicate.side_effect = fail_compile_and_touch_class

    with mock.patch(
        "interpreter.core.computer.terminal.languages.java.subprocess.Popen",
        return_value=compile_proc,
    ):
        list(java.run(_VALID_CLASS))

    assert not (tmp_path / "Hello.java").exists()
    assert not (tmp_path / "Hello.class").exists()


def test_java_run_success_cleans_up_after_execution(tmp_path, monkeypatch):
    """Java.run() deletes generated .java and .class files after successful execution."""
    import queue
    import threading

    monkeypatch.chdir(tmp_path)
    java = Java()
    java.output_queue = queue.Queue()
    java.done = threading.Event()
    java.done.set()

    compile_proc = mock.Mock()
    compile_proc.communicate.return_value = ("", "")
    compile_proc.returncode = 0

    run_proc = mock.Mock()
    run_proc.stdout = mock.Mock()
    run_proc.stderr = mock.Mock()
    run_proc.wait.return_value = None

    def popen_side_effect(cmd, **kwargs):
        if cmd[0] == "javac":
            (tmp_path / "Hello.class").write_bytes(b"fake")
            return compile_proc
        return run_proc

    with mock.patch(
        "interpreter.core.computer.terminal.languages.java.subprocess.Popen",
        side_effect=popen_side_effect,
    ):
        with mock.patch(
            "interpreter.core.computer.terminal.languages.java.threading.Thread"
        ) as thread_cls:
            thread_cls.return_value.start = mock.Mock()
            thread_cls.return_value.join = mock.Mock()
            list(java.run(_VALID_CLASS))

    assert not (tmp_path / "Hello.java").exists()
    assert not (tmp_path / "Hello.class").exists()


def test_java_line_postprocessor_strips_whitespace():
    """Java line_postprocessor trims leading and trailing whitespace from output lines."""
    java = Java()
    assert java.line_postprocessor("  output  ") == "output"


def test_java_detect_active_line():
    """Java detect_active_line() parses ##active_lineN## markers and ignores plain text."""
    java = Java()
    assert java.detect_active_line("##active_line2##") == 2
    assert java.detect_active_line("plain") is None
