import shutil
import subprocess
import sys
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[1] / "installers" / "oi-linux-installer.sh"


def _pip_install_line() -> str:
    """Return the installer line that runs the pyenv-wrapped pip install."""
    for line in INSTALLER.read_text(encoding="utf-8").splitlines():
        if "pip install open-interpreter" in line and "pyenv_root" in line:
            return line
    raise AssertionError("installer no longer contains a pyenv pip install line")


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the Linux installer targets Unix; Windows CI's bash is the unrelated WSL launcher",
)
def test_linux_installer_pins_the_selected_python_for_pip():
    """The pip install must see the installer's python_version, not pyenv's default.

    `pyenv shell` only prints code to eval, so a script that calls it directly
    leaves the next `pyenv exec` on the global default interpreter. Running the
    real pip line with a shell-function stub reproduces that: the stub reports
    the version pyenv would use, which must be the 3.11.7 the script installed.
    The stub is a shell function so the check never executes a script file, and
    the test skips Windows, where `bash` is the WSL launcher rather than a POSIX
    shell.
    """
    stub = 'pyenv_root() { echo "version=${PYENV_VERSION:-unset}"; echo "args=$*"; }'
    line = _pip_install_line()
    command = f'{stub}; pyenv_root=pyenv_root; python_version="3.11.7"; {line}'
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "version=3.11.7",
        "args=exec",
        "pip",
        "install",
        "open-interpreter",
        "--break-system-packages",
    ]
