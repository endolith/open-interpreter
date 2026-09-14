"""Language runtimes must not outlive the interpreter's process.

Terminal.stop() only interrupts running code; only Terminal.terminate() kills
the Jupyter kernel and shell subprocesses, and nothing called it at exit, so
every CLI session and every test run left an ipykernel_launcher behind, each
holding its ZMQ sockets.
"""

import json
import subprocess
import sys
import time

import psutil

# Writes the spawned child PIDs to the path in argv[1]. Output goes to a file,
# not a pipe: a leaked child would hold a pipe open and hang the test instead
# of failing it.
SCRIPT = """
import json, sys
import psutil
from interpreter import OpenInterpreter
i = OpenInterpreter()
i.offline = True
i.disable_telemetry = True
i.computer.run("python", "print(1)")
children = [p.pid for p in psutil.Process().children(recursive=True)]
open(sys.argv[1], "w").write(json.dumps(children))
sys.exit(0)
"""


def _alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_runtimes_are_terminated_at_interpreter_exit(tmp_path):
    """After a normal exit, every child the terminal spawned is gone within five seconds."""
    out = tmp_path / "children.json"
    log = tmp_path / "log.txt"
    with open(log, "w") as logfile:
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT, str(out)],
            stdin=subprocess.DEVNULL,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            timeout=180,
        )

    assert result.returncode == 0, log.read_text()
    children = json.loads(out.read_text())
    assert children, "the script should have spawned a kernel"

    deadline = time.time() + 5
    while time.time() < deadline and any(_alive(pid) for pid in children):
        time.sleep(0.2)

    survivors = [pid for pid in children if _alive(pid)]
    for pid in survivors:  # never leave them behind, even when the test fails
        try:
            psutil.Process(pid).kill()
        except psutil.NoSuchProcess:
            pass
    assert not survivors, f"runtimes outlived the interpreter: {survivors}"


def test_terminate_is_idempotent():
    """Calling terminate() twice does not raise.

    The atexit hook runs after any explicit terminate() a caller made, so the
    second call must be harmless or exit would fail with a traceback.
    """
    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    interpreter.offline = True
    interpreter.disable_telemetry = True
    interpreter.computer.run("python", "print(1)")

    interpreter.computer.terminal.terminate()
    interpreter.computer.terminal.terminate()
