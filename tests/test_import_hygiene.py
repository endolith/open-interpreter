import subprocess
import sys

THIRD_PARTY = ("inquirer", "yaml", "requests", "platformdirs", "send2trash")

_SCRIPT = """
import sys

sys.argv = ["interpreter", {flag!r}]
import interpreter.terminal_interface.start_terminal_interface  # noqa: F401
print(",".join(name for name in {modules!r} if name in sys.modules))
"""


def _loaded_third_party(flag):
    """Return the third-party modules loaded by importing the CLI entry module."""
    script = _SCRIPT.format(flag=flag, modules=THIRD_PARTY)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_help_import_path_loads_no_third_party_packages():
    """Importing the CLI entry module under --help loads no third-party packages.

    --help and --version exit inside argparse before any flag handling runs, so
    the packages only some flags need must stay at their point of use. Loading
    them at import time makes every invocation pay for packages the run may
    never touch.
    """
    assert _loaded_third_party("--help") == ""


def test_version_import_path_loads_no_third_party_packages():
    """--version, the other immediate-exit flag, gets the same guarantee."""
    assert _loaded_third_party("--version") == ""


_VERSION_SCRIPT = """
import sys

sys.argv = ["interpreter", "--version"]
from interpreter.terminal_interface.start_terminal_interface import main

try:
    main()
except SystemExit as exc:
    code = exc.code
else:
    code = 0
loaded = [name for name in {modules!r} if name in sys.modules]
print(f"EXIT={{code}} LOADED={{','.join(loaded)}}")
"""


def test_version_flag_exits_cleanly_without_third_party_modules():
    """`main()` under --version exits 0 without loading the deferred packages.

    The fast-exit stub must satisfy every attribute the argument definitions
    read before argparse runs; missing one turned --version into an
    AttributeError instead of a version print.
    """
    script = _VERSION_SCRIPT.format(modules=THIRD_PARTY)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "EXIT=0 LOADED=" in result.stdout
    assert "Open Interpreter" in result.stdout
